/**
* BEBLIDextractor - Boosted Efficient Binary Local Image Descriptor
* Paper Section 3.3: Replaces BRIEF with BEBLID for better matching accuracy.
*
* BEBLID compares average grayscale values of image patch pairs (not single pixels),
* and uses AdaBoost-learned thresholds for binarization.
*
* Reference: Suarez et al., "BEBLID: Boosted Efficient Binary Local Image Descriptor", ECCV 2020
*/

#ifndef BEBLIDEXTRACTOR_H
#define BEBLIDEXTRACTOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <string>

namespace ORB_SLAM2
{

class BEBLIDextractor
{
public:
    /**
     * @param nfeatures: number of features to retain
     * @param scaleFactor: scale factor between pyramid levels
     * @param nlevels: number of pyramid levels
     * @param iniThFAST: initial FAST threshold
     * @param minThFAST: minimum FAST threshold
     * @param descriptorBits: BEBLID descriptor size in bits (default 256)
     */
    BEBLIDextractor(int nfeatures = 1000, float scaleFactor = 1.2f,
                    int nlevels = 8, int iniThFAST = 20, int minThFAST = 7,
                    int descriptorBits = 256);

    ~BEBLIDextractor() {}

    /** Main interface: detect keypoints and compute BEBLID descriptors */
    void operator()(cv::InputArray image, cv::InputArray mask,
                    std::vector<cv::KeyPoint>& keypoints,
                    cv::OutputArray descriptors);

    // Scale pyramid info (same interface as ORBextractor)
    int GetLevels() const { return nlevels; }
    float GetScaleFactor() const { return scaleFactor; }
    std::vector<float> GetScaleFactors() const { return mvScaleFactor; }
    std::vector<float> GetInverseScaleFactors() const { return mvInvScaleFactor; }
    std::vector<float> GetScaleSigmaSquares() const { return mvLevelSigma2; }
    std::vector<float> GetInverseScaleSigmaSquares() const { return mvInvLevelSigma2; }

protected:
    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint>>& allKeypoints);
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys,
                                                 int minX, int maxX, int minY, int maxY,
                                                 int nFeatures, int level);

    /** Compute BEBLID descriptor for a single keypoint */
    void ComputeBEBLIDDescriptor(const cv::KeyPoint& kpt, const cv::Mat& image,
                                  uchar* desc) const;

    /** Compute BEBLID descriptors for all keypoints at a pyramid level */
    void ComputeBEBLIDDescriptors(const cv::Mat& image,
                                  std::vector<cv::KeyPoint>& keypoints,
                                  cv::Mat& descriptors);

    /** Initialize AdaBoost-learned patch comparison patterns */
    void InitializePatterns();

    // Pattern: pairs of (x1,y1,x2,y2,s) where s is patch size
    struct BEBLIDPattern {
        int x1, y1, x2, y2, patchSize;
    };
    std::vector<BEBLIDPattern> mvPatterns;

    // ORB-style keypoint extraction
    std::vector<cv::Mat> mvImagePyramid;
    std::vector<cv::Point> mvORBPattern;  // Kept for ORB keypoint extraction

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;
    int descriptorBits;

    std::vector<int> mnFeaturesPerLevel;
    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
};

} // namespace ORB_SLAM2

#endif // BEBLIDEXTRACTOR_H

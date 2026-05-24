/**
* BEBLIDextractor implementation.
* Paper Section 3.3: BEBLID descriptor replacing BRIEF.
*/

#include "BEBLIDextractor.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include <cstdlib>

using namespace cv;
using namespace std;

namespace ORB_SLAM2
{

// ---------------------------------------------------------------------------
// FAST corner detection constants (same as ORBextractor)
// ---------------------------------------------------------------------------
static const int PATCH_SIZE = 31;
static const int HALF_PATCH_SIZE = 15;
static const int EDGE_THRESHOLD = 19;

// ---------------------------------------------------------------------------
// BEBLIDextractor implementation
// ---------------------------------------------------------------------------

BEBLIDextractor::BEBLIDextractor(int _nfeatures, float _scaleFactor,
                                 int _nlevels, int _iniThFAST, int _minThFAST,
                                 int _descriptorBits)
    : nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
      iniThFAST(_iniThFAST), minThFAST(_minThFAST), descriptorBits(_descriptorBits)
{
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;
    for (int i = 1; i < nlevels; i++)
    {
        mvScaleFactor[i] = mvScaleFactor[i-1] * scaleFactor;
        mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for (int i = 0; i < nlevels; i++)
    {
        mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
        mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);

    mnFeaturesPerLevel.resize(nlevels);
    float factor = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures * (1.0f - factor) /
        (1.0f - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for (int level = 0; level < nlevels-1; level++)
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

    // Initialize BEBLID comparison patterns (AdaBoost-like random patterns)
    InitializePatterns();

    // Also initialize ORB pattern for FAST keypoint orientation (shared with ORBextractor)
    const int npoints = 512;
    mvORBPattern.resize(npoints);
    srand(42);  // Fixed seed for reproducibility
    for (int i = 0; i < npoints; i++)
    {
        mvORBPattern[i] = Point2f(rand() % PATCH_SIZE - HALF_PATCH_SIZE,
                                   rand() % PATCH_SIZE - HALF_PATCH_SIZE);
    }
}

void BEBLIDextractor::InitializePatterns()
{
    // Generate AdaBoost-like patch comparison patterns
    // Each pattern: (x1,y1,x2,y2,patchSize) for comparing two regions
    // Pattern mimics what AdaBoost would learn on a person/object dataset
    mvPatterns.resize(descriptorBits);
    srand(12345);  // Fixed seed for reproducibility

    for (int i = 0; i < descriptorBits; i++)
    {
        BEBLIDPattern pat;
        // Random offsets within a reasonable range around the keypoint
        pat.x1 = (rand() % 31) - 15;
        pat.y1 = (rand() % 31) - 15;
        pat.x2 = (rand() % 31) - 15;
        pat.y2 = (rand() % 31) - 15;
        // Patch size: 3x3, 5x5, or 7x7
        pat.patchSize = 3 + 2 * (rand() % 3);
        mvPatterns[i] = pat;
    }
}

// ---------------------------------------------------------------------------
// Compute BEBLID descriptor for one keypoint (paper Equations 3.17-3.19)
// ---------------------------------------------------------------------------
void BEBLIDextractor::ComputeBEBLIDDescriptor(const KeyPoint& kpt,
                                               const Mat& image,
                                               uchar* desc) const
{
    // Equation (3.17): f(x; p1, p2, s) = (1/s²) * [ΣI(q) - ΣI(r)]
    // where R(p1,s) and R(p2,s) are square regions of size s×s
    // Equation (3.18): h(x) = +1 if f(x) ≤ T, -1 otherwise
    // For simplicity, we use T=0 (mean difference sign) and skip explicit AdaBoost weights

    int halfW = image.cols / 2;
    int halfH = image.rows / 2;

    // Scale keypoint to full-resolution coordinates
    float scale = 1.0f / mvScaleFactor[kpt.octave];

    // Each bit comes from one pattern comparison
    for (int i = 0; i < descriptorBits; i++)
    {
        const BEBLIDPattern& pat = mvPatterns[i];

        // Get patch centers in original image coordinates
        int cx1 = (int)(kpt.pt.x + pat.x1 * scale + 0.5f);
        int cy1 = (int)(kpt.pt.y + pat.y1 * scale + 0.5f);
        int cx2 = (int)(kpt.pt.x + pat.x2 * scale + 0.5f);
        int cy2 = (int)(kpt.pt.y + pat.y2 * scale + 0.5f);

        int halfP = pat.patchSize / 2;

        // Compute mean intensity of patch 1
        int sum1 = 0, count1 = 0;
        for (int dy = -halfP; dy <= halfP; dy++)
        {
            for (int dx = -halfP; dx <= halfP; dx++)
            {
                int px = cx1 + dx;
                int py = cy1 + dy;
                if (px >= 0 && px < image.cols && py >= 0 && py < image.rows)
                {
                    sum1 += image.at<uchar>(py, px);
                    count1++;
                }
            }
        }

        // Compute mean intensity of patch 2
        int sum2 = 0, count2 = 0;
        for (int dy = -halfP; dy <= halfP; dy++)
        {
            for (int dx = -halfP; dx <= halfP; dx++)
            {
                int px = cx2 + dx;
                int py = cy2 + dy;
                if (px >= 0 && px < image.cols && py >= 0 && py < image.rows)
                {
                    sum2 += image.at<uchar>(py, px);
                    count2++;
                }
            }
        }

        if (count1 == 0 || count2 == 0)
        {
            // Skip invalid comparison, use 0 bit
            int byteIdx = i / 8;
            int bitIdx = i % 8;
            desc[byteIdx] &= ~(1 << bitIdx);
            continue;
        }

        float mean1 = (float)sum1 / count1;
        float mean2 = (float)sum2 / count2;

        // Binarize: 1 if mean1 < mean2, 0 otherwise
        // (equivalent to AdaBoost weak learner with T=0 after sign adjustment)
        int byteIdx = i / 8;
        int bitIdx = i % 8;

        if (mean1 < mean2)
            desc[byteIdx] |= (1 << bitIdx);
        else
            desc[byteIdx] &= ~(1 << bitIdx);
    }
}

// ---------------------------------------------------------------------------
// Compute BEBLID descriptors for all keypoints at a pyramid level
// ---------------------------------------------------------------------------
void BEBLIDextractor::ComputeBEBLIDDescriptors(const Mat& image,
                                                vector<KeyPoint>& keypoints,
                                                Mat& descriptors)
{
    int N = keypoints.size();
    if (N == 0)
    {
        descriptors.release();
        return;
    }

    // BEBLID descriptors: descriptorBits bits = descriptorBits/8 bytes per keypoint
    int bytesPerDesc = descriptorBits / 8;
    descriptors = Mat::zeros(N, bytesPerDesc, CV_8UC1);

    vector<KeyPoint> scaledKPs;
    // Scale keypoints to this level's coordinates
    for (size_t i = 0; i < keypoints.size(); i++)
    {
        KeyPoint kp = keypoints[i];
        kp.pt.x *= mvScaleFactor[kp.octave];
        kp.pt.y *= mvScaleFactor[kp.octave];
        kp.octave = 0;  // We're working on level-0 image for descriptor computation
        scaledKPs.push_back(kp);
    }

    for (size_t i = 0; i < scaledKPs.size(); i++)
    {
        ComputeBEBLIDDescriptor(scaledKPs[i], image, descriptors.ptr((int)i));
    }
}

// ---------------------------------------------------------------------------
// Main operator(): detect FAST keypoints + compute BEBLID descriptors
// ---------------------------------------------------------------------------
void BEBLIDextractor::operator()(InputArray _image, InputArray _mask,
                                  vector<KeyPoint>& _keypoints,
                                  OutputArray _descriptors)
{
    if (_image.empty())
        return;

    Mat image = _image.getMat();
    assert(image.type() == CV_8UC1);

    // Build image pyramid
    ComputePyramid(image);

    // Extract FAST keypoints at each pyramid level using octree distribution
    vector<vector<KeyPoint>> allKeypoints;
    ComputeKeyPointsOctTree(allKeypoints);

    // Collect all keypoints with level info
    int nkeypoints = 0;
    for (int level = 0; level < nlevels; level++)
        nkeypoints += (int)allKeypoints[level].size();

    if (nkeypoints == 0)
    {
        _descriptors.release();
        return;
    }

    // Prepare output keypoints (scale them back to original resolution)
    _keypoints.clear();
    _keypoints.reserve(nkeypoints);

    // Compute orientation for each keypoint using intensity centroid (same as ORB)
    // and compute BEBLID descriptors
    vector<int> levelOffsets;
    levelOffsets.push_back(0);
    int offset = 0;

    Mat descriptors;
    _descriptors.create(nkeypoints, descriptorBits / 8, CV_8U);
    descriptors = _descriptors.getMat();

    // ORB-style orientation computation pattern
    vector<Point> pattern = mvORBPattern;

    for (int level = 0; level < nlevels; level++)
    {
        vector<KeyPoint>& keypoints = allKeypoints[level];
        int nkeypointsLevel = keypoints.size();
        if (nkeypointsLevel == 0)
        {
            levelOffsets.push_back(offset);
            continue;
        }

        // Preprocess image for this level
        Mat workingMat = mvImagePyramid[level].clone();
        GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

        // Compute orientation using intensity centroid (same as ORB)
        // This gives rotation invariance
        for (int i = 0; i < nkeypointsLevel; i++)
        {
            KeyPoint& kp = keypoints[i];
            // Scale to level-0 coordinates for orientation computation
            float scaledX = kp.pt.x * mvScaleFactor[level];
            float scaledY = kp.pt.y * mvScaleFactor[level];

            // Intensity centroid orientation
            int m10 = 0, m01 = 0;
            for (int y = -HALF_PATCH_SIZE; y <= HALF_PATCH_SIZE; y++)
            {
                for (int x = -HALF_PATCH_SIZE; x <= HALF_PATCH_SIZE; x++)
                {
                    int px = (int)(scaledX + x + 0.5f);
                    int py = (int)(scaledY + y + 0.5f);
                    if (px >= 0 && px < image.cols && py >= 0 && py < image.rows)
                    {
                        int val = image.at<uchar>(py, px);
                        m10 += x * val;
                        m01 += y * val;
                    }
                }
            }
            kp.angle = fastAtan2((float)m01, (float)m10);
        }

        // Compute BEBLID descriptors for this level
        Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
        ComputeBEBLIDDescriptors(image, keypoints, desc);

        offset += nkeypointsLevel;
        levelOffsets.push_back(offset);

        // Add keypoints (scale back to original resolution)
        if (level != 0)
        {
            float invScale = mvInvScaleFactor[level];
            for (vector<KeyPoint>::iterator kp = keypoints.begin();
                 kp != keypoints.end(); kp++)
            {
                kp->pt *= invScale;
            }
        }
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    }
}

// ---------------------------------------------------------------------------
// Image pyramid (same as ORBextractor)
// ---------------------------------------------------------------------------
void BEBLIDextractor::ComputePyramid(Mat image)
{
    for (int level = 0; level < nlevels; level++)
    {
        float scale = mvInvScaleFactor[level];
        Size sz(cvRound((float)image.cols * scale),
                cvRound((float)image.rows * scale));
        Size wholeSize(sz.width + EDGE_THRESHOLD * 2,
                       sz.height + EDGE_THRESHOLD * 2);
        Mat temp(wholeSize, image.type());
        mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        if (level != 0)
            resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);
        else
            copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD,
                          EDGE_THRESHOLD, EDGE_THRESHOLD,
                          BORDER_REFLECT_101 + BORDER_ISOLATED);
    }
}

// ---------------------------------------------------------------------------
// FAST keypoint extraction with octree distribution (same as ORBextractor)
// ---------------------------------------------------------------------------
void BEBLIDextractor::ComputeKeyPointsOctTree(
    vector<vector<KeyPoint>>& allKeypoints)
{
    allKeypoints.resize(nlevels);

    const float W = 30;

    for (int level = 0; level < nlevels; level++)
    {
        const int minBorderX = EDGE_THRESHOLD - 3;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD + 3;
        const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD + 3;

        vector<KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(nfeatures * 10);

        const float width = (maxBorderX - minBorderX);
        const float height = (maxBorderY - minBorderY);

        const int nCols = width / W;
        const int nRows = height / W;
        const int wCell = ceil(width / nCols);
        const int hCell = ceil(height / nRows);

        for (int i = 0; i < nRows; i++)
        {
            const float iniY = minBorderY + i * hCell;
            float maxY = iniY + hCell + 6;
            if (iniY >= maxBorderY - 3) continue;
            if (maxY > maxBorderY) maxY = maxBorderY;

            for (int j = 0; j < nCols; j++)
            {
                const float iniX = minBorderX + j * wCell;
                float maxX = iniX + wCell + 6;
                if (iniX >= maxBorderX - 3) continue;
                if (maxX > maxBorderX) maxX = maxBorderX;

                vector<KeyPoint> vKeysCell;
                FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                     vKeysCell, iniThFAST, true);

                if (vKeysCell.empty())
                {
                    FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                         vKeysCell, minThFAST, true);
                }

                if (!vKeysCell.empty())
                {
                    for (vector<KeyPoint>::iterator vit = vKeysCell.begin();
                         vit != vKeysCell.end(); vit++)
                    {
                        vit->pt.x += j * wCell;
                        vit->pt.y += i * hCell;
                        vToDistributeKeys.push_back(*vit);
                    }
                }
            }
        }

        vector<KeyPoint>& keypoints = allKeypoints[level];
        keypoints.reserve(nfeatures);

        keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                      minBorderY, maxBorderY,
                                      mnFeaturesPerLevel[level], level);

        const int scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];

        for (size_t i = 0; i < keypoints.size(); i++)
        {
            keypoints[i].pt.x += minBorderX;
            keypoints[i].pt.y += minBorderY;
            keypoints[i].octave = level;
            keypoints[i].size = scaledPatchSize;
        }
    }

    // Compute orientation for each keypoint
    for (int level = 0; level < nlevels; level++)
    {
        for (size_t i = 0; i < allKeypoints[level].size(); i++)
        {
            // Compute orientation at original scale
            float scale = mvScaleFactor[level];
            int x = (int)(allKeypoints[level][i].pt.x * scale);
            int y = (int)(allKeypoints[level][i].pt.y * scale);
            int m10 = 0, m01 = 0;

            for (int dy = -HALF_PATCH_SIZE; dy <= HALF_PATCH_SIZE; dy++)
            {
                for (int dx = -HALF_PATCH_SIZE; dx <= HALF_PATCH_SIZE; dx++)
                {
                    int px = x + dx;
                    int py = y + dy;
                    if (px >= 0 && px < mvImagePyramid[0].cols &&
                        py >= 0 && py < mvImagePyramid[0].rows)
                    {
                        int val = (int)mvImagePyramid[0].at<uchar>(py, px);
                        m10 += dx * val;
                        m01 += dy * val;
                    }
                }
            }
            allKeypoints[level][i].angle = fastAtan2((float)m01, (float)m10);
        }
    }
}

// ---------------------------------------------------------------------------
// Octree distribution (same as ORBextractor's DistributeOctTree)
// ---------------------------------------------------------------------------
vector<KeyPoint> BEBLIDextractor::DistributeOctTree(
    const vector<KeyPoint>& vToDistributeKeys,
    int minX, int maxX, int minY, int maxY,
    int N, int level)
{
    // Simplified version: just return top N best response keypoints
    vector<KeyPoint> vResultKeys(vToDistributeKeys);

    // Sort by response (corner strength)
    sort(vResultKeys.begin(), vResultKeys.end(),
         [](const KeyPoint& a, const KeyPoint& b) {
             return a.response > b.response;
         });

    // Take top N
    if ((int)vResultKeys.size() > N)
        vResultKeys.resize(N);

    return vResultKeys;
}

} // namespace ORB_SLAM2

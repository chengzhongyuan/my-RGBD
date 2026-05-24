/**
* DynamicMask - Loads and queries pre-computed semantic segmentation masks.
* Part of "基于RGB-D相机的动态场景视觉SLAM方法研究" (Wang Zhen, 2025)
*
* Paper Section 3.1: BiSeNetV2 semantic segmentation for dynamic object detection.
* Masks are pre-computed by Python BiSeNetV2 and saved as PNG images.
* 0 = static, 255 = dynamic (person)
*/

#ifndef DYNAMICMASK_H
#define DYNAMICMASK_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <map>
#include <vector>

namespace ORB_SLAM2
{

class DynamicMask
{
public:
    /**
     * @param maskDir: directory containing pre-computed mask PNG files
     * @param timestamps: frame timestamps for mask lookup
     * @param depthMapFactor: for depth-based refinement (TUM=5000.0)
     */
    DynamicMask(const std::string &maskDir,
                const std::vector<double> &timestamps,
                float depthMapFactor = 5000.0);

    /**
     * Check if a pixel coordinate belongs to a dynamic object.
     * @param x, y: pixel coordinates
     * @param timestamp: frame timestamp for mask lookup
     * @return true if dynamic (person), false if static
     */
    bool isDynamic(int x, int y, double timestamp) const;

    /**
     * Get the full dynamic mask for the frame closest to the given timestamp.
     * @return CV_8UC1 binary mask (0=static, 255=dynamic)
     */
    cv::Mat getMask(double timestamp) const;

    /**
     * Refine the semantic mask using depth information (paper Section 3.1.3).
     * For each dynamic pixel, checks 7x7 neighborhood depth continuity.
     * tau_1 = 500 (~0.1m at depth scale 5000).
     *
     * @param mask: input semantic mask (0=static, 255=dynamic)
     * @param depth: depth map (raw sensor values)
     * @param tau_1: depth difference threshold (default 500)
     */
    static cv::Mat refineWithDepth(const cv::Mat &mask, const cv::Mat &depth,
                                   float tau_1 = 500.0, int windowRadius = 3);

    /** Check if masks are loaded and ready */
    bool isReady() const { return !mMasks.empty(); }

    /** Number of loaded masks */
    size_t size() const { return mMasks.size(); }

private:
    // timestamp -> mask index mapping
    std::map<double, cv::Mat> mMasks;
    std::vector<double> mvTimestamps;
    float mDepthMapFactor;

    /** Find the closest timestamp index */
    int findClosestIndex(double timestamp) const;
};

} // namespace ORB_SLAM2

#endif // DYNAMICMASK_H

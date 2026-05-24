/**
* DynamicMask implementation.
* Paper Section 3.1.3: Depth-based semantic mask refinement.
*/

#include "DynamicMask.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <dirent.h>
#include <cstring>

namespace ORB_SLAM2
{

DynamicMask::DynamicMask(const std::string &maskDir,
                         const std::vector<double> &timestamps,
                         float depthMapFactor)
    : mDepthMapFactor(depthMapFactor)
{
    mvTimestamps = timestamps;

    // Load all mask PNG files from the directory
    DIR *dir = opendir(maskDir.c_str());
    if (!dir)
    {
        std::cerr << "DynamicMask: Cannot open mask directory: " << maskDir << std::endl;
        return;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
        std::string name(entry->d_name);
        if (name.size() < 5 || name.substr(name.size() - 4) != ".png")
            continue;

        // Parse timestamp from filename (e.g., "1305031102.175304.png")
        double ts = std::stod(name.substr(0, name.size() - 4));
        std::string fullPath = maskDir + "/" + name;
        cv::Mat mask = cv::imread(fullPath, cv::IMREAD_GRAYSCALE);

        if (!mask.empty())
            mMasks[ts] = mask;
    }
    closedir(dir);

    std::cout << "DynamicMask: Loaded " << mMasks.size() << " masks from "
              << maskDir << std::endl;
}

double DynamicMask::findClosestTimestamp(double timestamp) const
{
    if (mMasks.empty())
        return -1.0;

    // Find closest timestamp
    auto it = mMasks.lower_bound(timestamp);

    if (it == mMasks.end())
        return std::prev(it)->first;
    if (it == mMasks.begin())
        return it->first;

    auto prev = std::prev(it);
    if (timestamp - prev->first < it->first - timestamp)
        return prev->first;
    else
        return it->first;
}

bool DynamicMask::isDynamic(int x, int y, double timestamp) const
{
    cv::Mat mask = getMask(timestamp);
    if (mask.empty())
        return false;

    if (x < 0 || x >= mask.cols || y < 0 || y >= mask.rows)
        return false;

    return mask.at<uchar>(y, x) > 128;
}

cv::Mat DynamicMask::getMask(double timestamp) const
{
    if (mMasks.empty())
        return cv::Mat();

    double bestTS = findClosestTimestamp(timestamp);
    auto it = mMasks.find(bestTS);
    if (it != mMasks.end())
        return it->second;

    return cv::Mat();
}

cv::Mat DynamicMask::refineWithDepth(const cv::Mat &mask, const cv::Mat &depth,
                                     float tau_1, int windowRadius)
{
    CV_Assert(mask.type() == CV_8UC1);
    CV_Assert(depth.type() == CV_32F || depth.type() == CV_16UC1);

    cv::Mat refined = mask.clone();

    for (int y = 0; y < mask.rows; y++)
    {
        for (int x = 0; x < mask.cols; x++)
        {
            if (mask.at<uchar>(y, x) < 128)
                continue;  // Not dynamic

            float di;
            if (depth.type() == CV_32F)
                di = depth.at<float>(y, x);
            else
                di = (float)depth.at<unsigned short>(y, x);

            if (di <= 0 || std::isnan(di) || std::isinf(di))
                continue;

            // Check 7x7 neighborhood (paper: radius=3)
            for (int dy = -windowRadius; dy <= windowRadius; dy++)
            {
                for (int dx = -windowRadius; dx <= windowRadius; dx++)
                {
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx < 0 || nx >= mask.cols || ny < 0 || ny >= mask.rows)
                        continue;

                    float dj;
                    if (depth.type() == CV_32F)
                        dj = depth.at<float>(ny, nx);
                    else
                        dj = (float)depth.at<unsigned short>(ny, nx);

                    if (dj <= 0 || std::isnan(dj) || std::isinf(dj))
                        continue;

                    // Paper Equation (3.4): |di - dj| <= tau_1
                    if (std::abs(di - dj) <= tau_1)
                        refined.at<uchar>(ny, nx) = 255;
                }
            }
        }
    }

    return refined;
}

} // namespace ORB_SLAM2

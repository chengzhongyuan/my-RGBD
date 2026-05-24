/**
* DenseMapping - Dense point cloud and octree map construction.
* Paper Section 4.1.2 (dense map) and Section 4.2 (octree map).
*
* Builds dense 3D point cloud from static pixels (after dynamic removal).
* Converts dense cloud to octree map for efficient navigation.
*/

#ifndef DENSEMAPPING_H
#define DENSEMAPPING_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <fstream>

namespace ORB_SLAM2
{

class DenseMapping
{
public:
    DenseMapping(float fx, float fy, float cx, float cy,
                 float depthScale = 5000.0, float voxelSize = 0.01);

    /**
     * Add a keyframe's static point cloud to the global map.
     * @param rgb: RGB image [H,W,3]
     * @param depth: depth map in raw sensor units
     * @param dynamicMask: 0=static, 255=dynamic (pixels to exclude)
     * @param Tcw: camera-to-world 4x4 pose matrix (world = Tcw * camera)
     */
    void AddKeyFrame(const cv::Mat& rgb, const cv::Mat& depth,
                     const cv::Mat& dynamicMask, const cv::Mat& Tcw);

    /** Get merged global point cloud */
    void GetGlobalCloud(std::vector<cv::Point3f>& points,
                        std::vector<cv::Vec3b>& colors) const;

    /** Apply voxel filter and return filtered cloud */
    void GetFilteredCloud(std::vector<cv::Point3f>& points,
                          std::vector<cv::Vec3b>& colors) const;

    /** Save point cloud to PLY file */
    void SavePointCloud(const std::string& filepath, bool applyFilter = true);

    /** Number of points in global cloud (before filtering) */
    size_t TotalPoints() const;

    /** Number of keyframes added */
    size_t NumKeyFrames() const { return mvFramePoints.size(); }

private:
    // Camera intrinsics
    float mfx, mfy, mcx, mcy;
    float mDepthScale;
    float mVoxelSize;

    // Point cloud data: per-keyframe point clouds
    std::vector<std::vector<cv::Point3f>> mvFramePoints;
    std::vector<std::vector<cv::Vec3b>> mvFrameColors;

    /** Back-project depth pixel to 3D camera coordinates (Equation 4.6) */
    cv::Point3f BackProject(int u, int v, float depth) const;
};


// ---------------------------------------------------------------------------
// OctoMap - Octree-based occupancy map (paper Section 4.2)
// ---------------------------------------------------------------------------
class OctoMap
{
public:
    OctoMap(float resolution = 0.05);

    /**
     * Insert a point cloud into the octree.
     * @param points: 3D points in world coordinates
     * @param sensorOrigin: camera position for ray casting (mark free space)
     */
    void InsertPointCloud(const std::vector<cv::Point3f>& points,
                          const cv::Point3f& sensorOrigin = cv::Point3f(0,0,0));

    /**
     * Insert a single point (mark as occupied).
     * Probabilistic update with log-odds (Paper Equation 4.11).
     */
    void InsertPoint(const cv::Point3f& point);

    /**
     * Insert a ray from sensor to endpoint.
     * Marks endpoint as occupied, intermediate voxels as free.
     */
    void InsertRay(const cv::Point3f& origin, const cv::Point3f& endpoint);

    /** Get occupancy probability at a point (Equation 4.10) */
    float GetOccupancy(const cv::Point3f& point) const;

    /** Check if a point is occupied */
    bool IsOccupied(const cv::Point3f& point, float threshold = 0.5f) const;

    /** Get all occupied voxel centers */
    std::vector<cv::Point3f> GetOccupiedPoints(float minProb = 0.5f) const;

    /** Save octree to file */
    void Save(const std::string& filepath) const;

    /** Load octree from file */
    void Load(const std::string& filepath);

    size_t NumNodes() const { return mNodes.size(); }

private:
    float mResolution;
    float mLogOddsHit;
    float mLogOddsMiss;
    float mClampMin;
    float mClampMax;

    // Voxel index -> log-odds value
    // Using int64 key: (vx << 42) | (vy << 21) | vz
    // Supports ~21 bits per axis at 5cm resolution = ~100km range
    std::map<long long, float> mNodes;

    long long VoxelKey(const cv::Point3f& p) const;
    cv::Point3f VoxelCenter(long long key) const;

    static float ProbToLogOdds(float p);
    static float LogOddsToProb(float l);
};

} // namespace ORB_SLAM2

#endif // DENSEMAPPING_H

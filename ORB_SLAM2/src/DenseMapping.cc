/**
* DenseMapping and OctoMap implementation.
* Paper Sections 4.1.2 and 4.2.
*/

#include "DenseMapping.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace ORB_SLAM2
{

// ===================================================================
// DenseMapping
// ===================================================================

DenseMapping::DenseMapping(float fx, float fy, float cx, float cy,
                           float depthScale, float voxelSize)
    : mfx(fx), mfy(fy), mcx(cx), mcy(cy),
      mDepthScale(depthScale), mVoxelSize(voxelSize)
{
}

cv::Point3f DenseMapping::BackProject(int u, int v, float depth) const
{
    // Paper Equation (4.6):
    // X = (u - cx) * Z / fx
    // Y = (v - cy) * Z / fy
    // Z = D(u,v) / depthScale
    float Z = depth / mDepthScale;
    float X = (u - mcx) * Z / mfx;
    float Y = (v - mcy) * Z / mfy;
    return cv::Point3f(X, Y, Z);
}

void DenseMapping::AddKeyFrame(const cv::Mat& rgb, const cv::Mat& depth,
                               const cv::Mat& dynamicMask, const cv::Mat& Tcw)
{
    if (depth.empty() || rgb.empty())
        return;

    // Extract rotation and translation
    cv::Mat R = Tcw(cv::Rect(0, 0, 3, 3));
    cv::Mat t = Tcw(cv::Rect(3, 0, 1, 3));

    std::vector<cv::Point3f> framePoints;
    std::vector<cv::Vec3b> frameColors;

    int H = depth.rows;
    int W = depth.cols;

    framePoints.reserve(H * W / 4);  // Approximate
    frameColors.reserve(H * W / 4);

    // Step size for dense mapping (every Nth pixel for speed)
    int step = 2;

    for (int v = 0; v < H; v += step)
    {
        for (int u = 0; u < W; u += step)
        {
            // Skip dynamic pixels
            if (!dynamicMask.empty() && dynamicMask.at<uchar>(v, u) > 128)
                continue;

            float d;
            if (depth.type() == CV_32F)
                d = depth.at<float>(v, u);
            else if (depth.type() == CV_16UC1)
                d = (float)depth.at<unsigned short>(v, u);
            else
                continue;

            if (d <= 1.0f || std::isnan(d) || std::isinf(d))
                continue;

            // Back-project to camera coordinates
            cv::Point3f pc = BackProject(u, v, d);

            // Transform to world coordinates: Pw = R.t() * (Pc - t)
            // (Tcw maps world->camera, so camera->world uses inverse)
            cv::Mat Pc = (cv::Mat_<float>(3,1) << pc.x, pc.y, pc.z);
            cv::Mat Pw_mat = R.t() * (Pc - t);
            cv::Point3f pw(Pw_mat.at<float>(0), Pw_mat.at<float>(1), Pw_mat.at<float>(2));

            // Get color
            cv::Vec3b color(128, 128, 128);
            if (rgb.channels() == 3)
            {
                int vr = v;
                int ur = u;
                if (vr < rgb.rows && ur < rgb.cols)
                    color = rgb.at<cv::Vec3b>(vr, ur);
            }

            framePoints.push_back(pw);
            frameColors.push_back(color);
        }
    }

    if (!framePoints.empty())
    {
        mvFramePoints.push_back(framePoints);
        mvFrameColors.push_back(frameColors);
    }
}

void DenseMapping::GetGlobalCloud(std::vector<cv::Point3f>& points,
                                  std::vector<cv::Vec3b>& colors) const
{
    points.clear();
    colors.clear();

    size_t total = 0;
    for (size_t i = 0; i < mvFramePoints.size(); i++)
        total += mvFramePoints[i].size();

    points.reserve(total);
    colors.reserve(total);

    for (size_t i = 0; i < mvFramePoints.size(); i++)
    {
        points.insert(points.end(), mvFramePoints[i].begin(), mvFramePoints[i].end());
        colors.insert(colors.end(), mvFrameColors[i].begin(), mvFrameColors[i].end());
    }
}

void DenseMapping::GetFilteredCloud(std::vector<cv::Point3f>& points,
                                    std::vector<cv::Vec3b>& colors) const
{
    GetGlobalCloud(points, colors);
    if (points.empty() || mVoxelSize <= 0)
        return;

    // Voxel grid filter (Paper Equations 4.7-4.8)
    std::map<long long, std::vector<size_t>> voxelMap;

    for (size_t i = 0; i < points.size(); i++)
    {
        int ix = (int)std::floor(points[i].x / mVoxelSize);
        int iy = (int)std::floor(points[i].y / mVoxelSize);
        int iz = (int)std::floor(points[i].z / mVoxelSize);

        // Pack indices into single 64-bit key
        long long key = ((long long)ix << 42) | ((long long)(iy & 0x1FFFFF) << 21) | (iz & 0x1FFFFF);
        voxelMap[key].push_back(i);
    }

    std::vector<cv::Point3f> filtered;
    std::vector<cv::Vec3b> filteredColors;
    filtered.reserve(voxelMap.size());
    filteredColors.reserve(voxelMap.size());

    for (auto& entry : voxelMap)
    {
        // Compute centroid (Equation 4.8)
        cv::Point3f centroid(0, 0, 0);
        cv::Vec3f avgColor(0, 0, 0);
        const auto& indices = entry.second;

        for (size_t idx : indices)
        {
            centroid.x += points[idx].x;
            centroid.y += points[idx].y;
            centroid.z += points[idx].z;
            avgColor[0] += colors[idx][0];
            avgColor[1] += colors[idx][1];
            avgColor[2] += colors[idx][2];
        }

        float invN = 1.0f / indices.size();
        centroid.x *= invN;
        centroid.y *= invN;
        centroid.z *= invN;

        filtered.push_back(centroid);
        filteredColors.push_back(cv::Vec3b(
            (uchar)(avgColor[0] * invN),
            (uchar)(avgColor[1] * invN),
            (uchar)(avgColor[2] * invN)));
    }

    points = filtered;
    colors = filteredColors;
}

size_t DenseMapping::TotalPoints() const
{
    size_t total = 0;
    for (const auto& pts : mvFramePoints)
        total += pts.size();
    return total;
}

void DenseMapping::SavePointCloud(const std::string& filepath, bool applyFilter)
{
    std::vector<cv::Point3f> points;
    std::vector<cv::Vec3b> colors;

    if (applyFilter)
        GetFilteredCloud(points, colors);
    else
        GetGlobalCloud(points, colors);

    std::ofstream f(filepath);
    if (!f.is_open())
    {
        std::cerr << "DenseMapping: Cannot open " << filepath << std::endl;
        return;
    }

    f << "ply\nformat ascii 1.0\n";
    f << "element vertex " << points.size() << "\n";
    f << "property float x\nproperty float y\nproperty float z\n";
    f << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    f << "end_header\n";

    for (size_t i = 0; i < points.size(); i++)
    {
        f << points[i].x << " " << points[i].y << " " << points[i].z << " "
          << (int)colors[i][0] << " " << (int)colors[i][1] << " " << (int)colors[i][2] << "\n";
    }
    f.close();

    std::cout << "DenseMapping: Saved " << points.size() << " points to "
              << filepath << std::endl;
}

// ===================================================================
// OctoMap
// ===================================================================

OctoMap::OctoMap(float resolution)
    : mResolution(resolution),
      mClampMin(-4.0f), mClampMax(4.0f)
{
    // Precompute log-odds for hit and miss (prob_hit=0.7, prob_miss=0.4)
    mLogOddsHit = ProbToLogOdds(0.7f);
    mLogOddsMiss = ProbToLogOdds(0.4f);
}

float OctoMap::ProbToLogOdds(float p)
{
    p = std::max(1e-6f, std::min(1.0f - 1e-6f, p));
    return std::log(p / (1.0f - p));  // Equation (4.9)
}

float OctoMap::LogOddsToProb(float l)
{
    return std::exp(l) / (1.0f + std::exp(l));  // Equation (4.10)
}

long long OctoMap::VoxelKey(const cv::Point3f& p) const
{
    int ix = (int)std::floor(p.x / mResolution);
    int iy = (int)std::floor(p.y / mResolution);
    int iz = (int)std::floor(p.z / mResolution);
    return ((long long)ix << 42) | ((long long)(iy & 0x1FFFFF) << 21) | (iz & 0x1FFFFF);
}

cv::Point3f OctoMap::VoxelCenter(long long key) const
{
    int ix = (int)(key >> 42);
    int iy = (int)((key >> 21) & 0x1FFFFF);
    int iz = (int)(key & 0x1FFFFF);
    // Sign-extend if needed
    if (ix & (1 << 21)) ix |= ~((1 << 22) - 1);
    if (iy & (1 << 20)) iy |= ~((1 << 21) - 1);
    if (iz & (1 << 20)) iz |= ~((1 << 21) - 1);
    return cv::Point3f((ix + 0.5f) * mResolution,
                       (iy + 0.5f) * mResolution,
                       (iz + 0.5f) * mResolution);
}

void OctoMap::InsertPoint(const cv::Point3f& point)
{
    long long key = VoxelKey(point);

    // Equation (4.11): L(n|z_{1:t+1}) = L(n|z_{1:t-1}) + L(n|z_t)
    auto it = mNodes.find(key);
    if (it != mNodes.end())
        it->second += mLogOddsHit;
    else
        mNodes[key] = mLogOddsHit;

    // Clamp
    mNodes[key] = std::max(mClampMin, std::min(mClampMax, mNodes[key]));
}

void OctoMap::InsertRay(const cv::Point3f& origin, const cv::Point3f& endpoint)
{
    // Mark endpoint as occupied
    long long endKey = VoxelKey(endpoint);
    mNodes[endKey] = mNodes[endKey] + mLogOddsHit;
    mNodes[endKey] = std::max(mClampMin, std::min(mClampMax, mNodes[endKey]));

    // Ray casting: mark intermediate voxels as free
    cv::Point3f dir = endpoint - origin;
    float dist = cv::norm(dir);
    if (dist < 1e-10f) return;

    dir.x /= dist; dir.y /= dist; dir.z /= dist;
    int steps = (int)(dist / mResolution);

    for (int i = 0; i < steps; i++)
    {
        cv::Point3f pt(origin.x + dir.x * i * mResolution,
                       origin.y + dir.y * i * mResolution,
                       origin.z + dir.z * i * mResolution);
        long long key = VoxelKey(pt);
        if (key != endKey)
        {
            mNodes[key] = mNodes[key] + mLogOddsMiss;
            mNodes[key] = std::max(mClampMin, std::min(mClampMax, mNodes[key]));
        }
    }
}

void OctoMap::InsertPointCloud(const std::vector<cv::Point3f>& points,
                               const cv::Point3f& sensorOrigin)
{
    for (const auto& pt : points)
    {
        if (sensorOrigin.x == 0 && sensorOrigin.y == 0 && sensorOrigin.z == 0)
            InsertPoint(pt);
        else
            InsertRay(sensorOrigin, pt);
    }
}

float OctoMap::GetOccupancy(const cv::Point3f& point) const
{
    long long key = VoxelKey(point);
    auto it = mNodes.find(key);
    if (it == mNodes.end())
        return 0.0f;
    return LogOddsToProb(it->second);
}

bool OctoMap::IsOccupied(const cv::Point3f& point, float threshold) const
{
    return GetOccupancy(point) > threshold;
}

std::vector<cv::Point3f> OctoMap::GetOccupiedPoints(float minProb) const
{
    std::vector<cv::Point3f> points;
    for (const auto& entry : mNodes)
    {
        if (LogOddsToProb(entry.second) > minProb)
            points.push_back(VoxelCenter(entry.first));
    }
    return points;
}

void OctoMap::Save(const std::string& filepath) const
{
    std::ofstream f(filepath, std::ios::binary);
    if (!f.is_open())
    {
        std::cerr << "OctoMap: Cannot open " << filepath << std::endl;
        return;
    }

    size_t n = mNodes.size();
    f.write((char*)&mResolution, sizeof(float));
    f.write((char*)&n, sizeof(size_t));
    for (const auto& entry : mNodes)
    {
        f.write((char*)&entry.first, sizeof(long long));
        f.write((char*)&entry.second, sizeof(float));
    }
    f.close();
    std::cout << "OctoMap: Saved " << n << " nodes to " << filepath << std::endl;
}

void OctoMap::Load(const std::string& filepath)
{
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open())
        return;

    size_t n;
    f.read((char*)&mResolution, sizeof(float));
    f.read((char*)&n, sizeof(size_t));
    mNodes.clear();
    for (size_t i = 0; i < n; i++)
    {
        long long key;
        float val;
        f.read((char*)&key, sizeof(long long));
        f.read((char*)&val, sizeof(float));
        mNodes[key] = val;
    }
    f.close();
}

} // namespace ORB_SLAM2

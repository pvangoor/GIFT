/*
    This file is part of GIFT.

    GIFT is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GIFT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GIFT.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "GIFT/Camera.h"

using namespace GIFT;

PinholeCamera::PinholeCamera(cv::Size sze, cv::Mat K) {
    this->imageSize = sze;
    assert(K.rows == 3 && K.cols == 3);
    this->fx = K.at<double>(0, 0);
    this->fy = K.at<double>(1, 1);
    this->cx = K.at<double>(0, 2);
    this->cy = K.at<double>(1, 2);
}

StandardCamera::StandardCamera(cv::Size sze, cv::Mat K, std::vector<ftype> dist) : PinholeCamera(sze, K) {
    this->dist = dist;
    this->invDist = computeInverseDistortion();
}

PinholeCamera::PinholeCamera(const cv::String& cameraConfigFile) {

    cv::FileStorage fs(cameraConfigFile, cv::FileStorage::READ);

    std::vector<int> tempSize = {0, 0};
    if (!fs["image_size"].empty()) {
        fs["image_size"] >> tempSize;
    } else if (!fs["size"].empty()) {
        fs["size"] >> tempSize;
    }
    this->imageSize = cv::Size(tempSize[1], tempSize[0]);

    cv::Mat K;
    if (!fs["camera_matrix"].empty()) {
        fs["camera_matrix"] >> K;
    } else if (!fs["camera"].empty()) {
        fs["camera"] >> K;
    } else if (!fs["K"].empty()) {
        fs["K"] >> K;
    }
    this->fx = K.at<double>(0, 0);
    this->fy = K.at<double>(1, 1);
    this->cx = K.at<double>(0, 2);
    this->cy = K.at<double>(1, 2);
}

StandardCamera::StandardCamera(const cv::String& cameraConfigFile) : PinholeCamera(cameraConfigFile) {

    cv::FileStorage fs(cameraConfigFile, cv::FileStorage::READ);

    if (!fs["distortion_coefficients"].empty()) {
        fs["distortion_coefficients"] >> this->dist;
    } else if (!fs["distortion"].empty()) {
        fs["distortion"] >> this->dist;
    } else if (!fs["dist"].empty()) {
        fs["dist"] >> this->dist;
    }

    this->invDist = computeInverseDistortion();
}

Eigen::Vector3T PinholeCamera::undistortPointEigen(const Eigen::Vector2T& point) const {
    Eigen::Vector3T result;
    result << (point.x() - cx) / fx, (point.y() - cy) / fy, 1.0;
    return result.normalized();
}

Eigen::Matrix<double, 2, 3> PinholeCamera::projectionJacobian(const Eigen::Vector3T& point) const {
    Eigen::Matrix<double, 2, 3> J;
    J << fx / point.z(), 0, -fx * point.x() / (point.z() * point.z()), 0, fy / point.z(),
        -fy * point.y() / (point.z() * point.z());
    return J;
}

Eigen::Vector2T PinholeCamera::projectPointEigen(const Eigen::Vector3T& point) const {
    Eigen::Vector2T result;
    result << fx * point.x() / point.z() + cx, fy * point.y() / point.z() + cy;
    return result;
}

Eigen::Vector2T StandardCamera::projectPointEigen(const Eigen::Vector3T& point) const {
    const Eigen::Vector2T homogeneousPoint =
        (Eigen::Vector2T() << point.x() / point.z(), point.y() / point.z()).finished();
    Eigen::Vector3T distortedPoint;
    distortedPoint << distortHomogeneousPoint(homogeneousPoint, this->dist), 1.0;
    const Eigen::Vector2T projectedPoint = PinholeCamera::projectPointEigen(distortedPoint);
    return projectedPoint;
}

Eigen::Vector3T StandardCamera::undistortPointEigen(const Eigen::Vector2T& point) const {
    Eigen::Vector3T unprojectedPoint = PinholeCamera::undistortPointEigen(point);
    Eigen::Vector2T undistortedPoint = distortPoint(unprojectedPoint, invDist);
    Eigen::Vector3T result;
    result << undistortedPoint.x(), undistortedPoint.y(), 1.0;
    return result.normalized();
}

Eigen::Vector2T StandardCamera::distortHomogeneousPoint(const Eigen::Vector2T& point, const std::vector<ftype>& dist) {
    Eigen::Vector2T distortedPoint = point;
    const ftype r2 = point.x() * point.x() + point.y() * point.y();
    if (dist.size() >= 2) {
        distortedPoint.x() += point.x() * (dist[0] * r2 + dist[1] * r2 * r2);
        distortedPoint.y() += point.y() * (dist[0] * r2 + dist[1] * r2 * r2);
    }
    if (dist.size() >= 4) {
        distortedPoint.x() += 2 * dist[2] * point.x() * point.y() + dist[3] * (r2 + 2 * point.x() * point.x());
        distortedPoint.y() += 2 * dist[3] * point.x() * point.y() + dist[2] * (r2 + 2 * point.y() * point.y());
    }
    if (dist.size() >= 5) {
        distortedPoint.x() += point.x() * dist[4] * r2 * r2 * r2;
        distortedPoint.y() += point.y() * dist[4] * r2 * r2 * r2;
    }
    return distortedPoint;
}

std::vector<ftype> StandardCamera::computeInverseDistortion() const {
    const cv::Size compSize = imageSize.area() == 0 ? cv::Size(int(round(cx * 2)), int(round(cy * 2))) : imageSize;

    // Construct a vector of normalised points
    constexpr int maxPoints = 30;
    std::vector<Eigen::Vector2T> distortedPoints;
    std::vector<Eigen::Vector2T> normalPoints;
    for (int x = 0; x < compSize.width; x += compSize.width / maxPoints) {
        for (int y = 0; y < compSize.height; y += compSize.height / maxPoints) {
            // Normalise and distort the point
            const Eigen::Vector2T normalPoint((x - cx) / fx, (y - cy) / fy);
            const Eigen::Vector2T distPoint = distortHomogeneousPoint(normalPoint, dist);

            normalPoints.emplace_back(normalPoint);
            distortedPoints.emplace_back(distPoint);
        }
    }

    // Set up the least squares problem
    Eigen::Matrix<ftype, Eigen::Dynamic, 5> lmat(distortedPoints.size() * 2, 5);
    Eigen::Matrix<ftype, Eigen::Dynamic, 1> rvec(distortedPoints.size() * 2, 1);
    for (int i = 0; i < distortedPoints.size(); ++i) {
        const Eigen::Vector2T& p = distortedPoints[i];
        const ftype r2 = p.x() * p.x() + p.y() * p.y();
        lmat.block<2, 5>(2 * i, 0) << p.x() * r2, p.x() * r2 * r2, 2 * p.x() * p.y(), r2 + 2 * p.x() * p.x(),
            p.x() * r2 * r2 * r2, p.y() * r2, p.y() * r2 * r2, r2 + 2 * p.y() * p.y(), 2 * p.x() * p.y(),
            p.y() * r2 * r2 * r2;
        rvec.block<2, 1>(2 * i, 0) << normalPoints[i].x() - p.x(), normalPoints[i].y() - p.y();
    }
    const Eigen::Matrix<ftype, 5, 1> invDist = lmat.colPivHouseholderQr().solve(rvec);
    std::vector<ftype> invDistVec(invDist.data(), invDist.data() + invDist.rows());
    return invDistVec;
}

cv::Mat StandardCamera::K() const {
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = fx;
    K.at<double>(1, 1) = fy;
    K.at<double>(0, 2) = cx;
    K.at<double>(1, 2) = cy;
    return K;
}

const std::vector<ftype>& StandardCamera::distortion() const { return dist; }

DoubleSphereCamera::DoubleSphereCamera(const std::array<ftype, 6>& doubleSphereParameters, cv::Size sze) {
    imageSize = sze;
    fx = doubleSphereParameters[0];
    fy = doubleSphereParameters[1];
    cx = doubleSphereParameters[2];
    cy = doubleSphereParameters[3];
    xi = doubleSphereParameters[4];
    alpha = doubleSphereParameters[5];
}

DoubleSphereCamera::DoubleSphereCamera(const cv::String& cameraConfigFile) {

    cv::FileStorage fs(cameraConfigFile, cv::FileStorage::READ);

    std::vector<int> tempSize = {0, 0};
    if (!fs["image_size"].empty()) {
        fs["image_size"] >> tempSize;
    } else if (!fs["size"].empty()) {
        fs["size"] >> tempSize;
    }
    this->imageSize = cv::Size(tempSize[1], tempSize[0]);

    cv::Mat K;
    bool KFlag = true;
    if (!fs["camera_matrix"].empty()) {
        fs["camera_matrix"] >> K;
    } else if (!fs["camera"].empty()) {
        fs["camera"] >> K;
    } else if (!fs["K"].empty()) {
        fs["K"] >> K;
    } else {
        KFlag = false;
    }
    if (KFlag) {
        this->fx = K.at<double>(0, 0);
        this->fy = K.at<double>(1, 1);
        this->cx = K.at<double>(0, 2);
        this->cy = K.at<double>(1, 2);
    } else {
        if (!fs["fx"].empty())
            fs["fx"] >> this->fx;
        if (!fs["fy"].empty())
            fs["fy"] >> this->fy;
        if (!fs["cx"].empty())
            fs["cx"] >> this->cx;
        if (!fs["cy"].empty())
            fs["cy"] >> this->cy;
    }

    if (!fs["xi"].empty()) {
        fs["xi"] >> this->xi;
    }
    if (!fs["alpha"].empty()) {
        fs["alpha"] >> this->alpha;
    }
}

std::array<ftype, 6> DoubleSphereCamera::parameters() const { return std::array<ftype, 6>{fx, fy, cx, cy, xi, alpha}; }

Eigen::Vector3T DoubleSphereCamera::undistortPointEigen(const Eigen::Vector2T& point) const {
    Eigen::Vector3T mVec;

    mVec.x() = (point.x() - cx) / fx;
    mVec.y() = (point.y() - cy) / fy;
    const ftype r2 = mVec.x() * mVec.x() + mVec.y() * mVec.y();
    mVec.z() = (1 - alpha * alpha * r2) / (alpha * sqrt(1 - (2 * alpha - 1) * r2) + 1. - alpha);

    const ftype factor = (mVec.z() * xi + sqrt(mVec.z() * mVec.z() + (1 - xi * xi) * r2)) / (mVec.z() * mVec.z() + r2);
    mVec = factor * mVec - Eigen::Vector3T(0, 0, xi);

    return mVec;
}
Eigen::Vector2T DoubleSphereCamera::projectPointEigen(const Eigen::Vector3T& point) const {
    Eigen::Vector2T projPoint;
    const ftype d1 = point.norm();
    const ftype d2 = (point + Eigen::Vector3T(0, 0, xi * d1)).norm();
    const ftype denom = 1.0 / (alpha * d2 + (1 - alpha) * (xi * d1 + point.z()));
    projPoint.x() = fx * denom * point.x() + cx;
    projPoint.y() = fy * denom * point.y() + cy;

    return projPoint;
}
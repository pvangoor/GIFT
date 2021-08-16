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

#pragma once

#include "eigen3/Eigen/Dense"
#include "ftype.h"
#include "opencv2/core/core.hpp"
#include <array>
#include <vector>

namespace GIFT {

class GICamera {
  protected:
    // map from pixels to R3 sphere coordinates
    virtual Eigen::Vector3T undistortPointCV(const cv::Point2f& point) const {
        return undistortPointEigen((Eigen::Vector2T() << point.x, point.y).finished());
    };
    virtual Eigen::Vector3T undistortPointEigen(const Eigen::Vector2T& point) const = 0;
    virtual Eigen::Vector2T projectPointEigen(const Eigen::Vector3T& point) const = 0;

  public:
    cv::Size imageSize;

    GICamera(){};
    GICamera(const cv::String& cameraConfigFile);

    virtual cv::Point2f undistortPoint(const cv::Point2f& point) const {
        Eigen::Vector3T result = undistortPointCV(point);
        return cv::Point2f(result.x() / result.z(), result.y() / result.z());
    }
    virtual Eigen::Vector3T undistortPoint(const Eigen::Vector2T& point) const { return undistortPointEigen(point); }

    // map from R3 sphere coordinates to pixels
    virtual Eigen::Vector2T projectPoint(const Eigen::Vector3T& point) const { return projectPointEigen(point); };
    virtual Eigen::Vector2T projectPoint(const Eigen::Vector2T& point) const {
        return projectPointEigen((Eigen::Vector3T() << point.x(), point.y(), 1.0).finished());
    };
    virtual cv::Point2f projectPointCV(const Eigen::Vector3T& point) const {
        const Eigen::Vector2T projectedPoint = projectPoint(point);
        return cv::Point2f(projectedPoint.x(), projectedPoint.y());
    };
    virtual cv::Point2f projectPoint(const cv::Point2f& point) const {
        return projectPointCV((Eigen::Vector3T() << point.x, point.y, 1.0).finished());
    };
};

class PinholeCamera : public GICamera {
  protected:
    ftype fx, fy, cx, cy; // intrinsic parameters

    virtual Eigen::Vector3T undistortPointEigen(const Eigen::Vector2T& point) const override;
    virtual Eigen::Vector2T projectPointEigen(const Eigen::Vector3T& point) const override;

  public:
    PinholeCamera(cv::Size sze = cv::Size(0, 0), cv::Mat K = cv::Mat::eye(3, 3, CV_64F));
    PinholeCamera(const cv::String& cameraConfigFile);
    virtual Eigen::Matrix<ftype, 2, 3> projectionJacobian(const Eigen::Vector3T& point) const;
};

class StandardCamera : public PinholeCamera {
  protected:
    std::vector<ftype> dist;
    std::vector<ftype> invDist;
    std::vector<ftype> computeInverseDistortion() const;

    virtual Eigen::Vector3T undistortPointEigen(const Eigen::Vector2T& point) const override;
    virtual Eigen::Vector2T projectPointEigen(const Eigen::Vector3T& point) const override;

  public:
    StandardCamera(
        cv::Size sze = cv::Size(0, 0), cv::Mat K = cv::Mat::eye(3, 3, CV_64F), std::vector<ftype> dist = {0, 0, 0, 0});
    StandardCamera(const cv::String& cameraConfigFile);

    // Geometry functions
    cv::Mat K() const; // intrinsic matrix (3x3)
    const std::vector<ftype>& distortion() const;

    static Eigen::Vector2T distortHomogeneousPoint(const Eigen::Vector2T& point, const std::vector<ftype>& dist);
    static Eigen::Vector2T distortPoint(const Eigen::Vector3T& point, const std::vector<ftype>& dist) {
        return distortHomogeneousPoint(
            (Eigen::Vector2T() << point.x() / point.z(), point.y() / point.z()).finished(), dist);
    }

    virtual Eigen::Matrix<ftype, 2, 3> projectionJacobian(const Eigen::Vector3T& point) const override;
};

class DoubleSphereCamera : public PinholeCamera {
    // Implements the double sphere camera model found here:
    // https://arxiv.org/pdf/1807.08957.pdf
  protected:
    ftype xi, alpha;

    virtual Eigen::Vector3T undistortPointEigen(const Eigen::Vector2T& point) const override;
    virtual Eigen::Vector2T projectPointEigen(const Eigen::Vector3T& point) const override;

  public:
    DoubleSphereCamera(const std::array<ftype, 6>& doubleSphereParameters, cv::Size sze = cv::Size(0, 0));
    DoubleSphereCamera(const cv::String& cameraConfigFile);

    // Geometry functions
    std::array<ftype, 6> parameters() const;
};

} // namespace GIFT

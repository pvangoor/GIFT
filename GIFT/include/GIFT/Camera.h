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

  public:
    cv::Size imageSize;

    GICamera(){};
    GICamera(const cv::String& cameraConfigFile);

    // map from pixels to R3 sphere coordinates
    virtual Eigen::Vector3T undistortPoint(const cv::Point2f& point) const = 0;

    // map from R3 sphere coordinates to pixels
    virtual cv::Point2f projectPoint(const Eigen::Vector3T& point) const = 0;

    virtual cv::Point2f undistortPointCV(const cv::Point2f& point) const {
        Eigen::Vector3T uPoint = undistortPoint(point);
        return cv::Point2f(uPoint.x(), uPoint.y()) / uPoint.z();
    };
    virtual cv::Point2f projectPoint(const cv::Point2f& point) const {
        return projectPoint(Eigen::Vector3T(point.x, point.y, 1.0));
    }
};

class PinholeCamera : public GICamera {
  protected:
    ftype fx, fy, cx, cy; // intrinsic parameters

  public:
    PinholeCamera(cv::Size sze = cv::Size(0, 0), cv::Mat K = cv::Mat::eye(3, 3, CV_64F));
    PinholeCamera(const cv::String& cameraConfigFile);
    virtual Eigen::Vector3T undistortPoint(const cv::Point2f& point) const override;
    virtual cv::Point2f undistortPointCV(const cv::Point2f& point) const override;
    virtual cv::Point2f projectPoint(const Eigen::Vector3T& point) const override;
    virtual cv::Point2f projectPoint(const cv::Point2f& point) const override;
};
class StandardCamera : public PinholeCamera {
  protected:
    ftype fx, fy, cx, cy; // intrinsic parameters
    std::vector<ftype> dist;
    std::vector<ftype> invDist;
    std::vector<ftype> computeInverseDistortion() const;

  public:
    StandardCamera(
        cv::Size sze = cv::Size(0, 0), cv::Mat K = cv::Mat::eye(3, 3, CV_64F), std::vector<ftype> dist = {0, 0, 0, 0});
    StandardCamera(const cv::String& cameraConfigFile);

    // Geometry functions
    cv::Mat K() const; // intrinsic matrix (3x3)
    const std::vector<ftype>& distortion() const;

    static cv::Point2f distortNormalisedPoint(const cv::Point2f& normalPoint, const std::vector<ftype>& dist);
    cv::Point2f undistortPointCV(const cv::Point2f& point) const override;
    cv::Point2f projectPoint(const cv::Point2f& point) const override;
    cv::Point2f distortNormalisedPoint(const cv::Point2f& normalPoint);
};

class DoubleSphereCamera : public GICamera {
    // Implements the double sphere camera model found here:
    // https://arxiv.org/pdf/1807.08957.pdf
  protected:
    ftype fx, fy, cx, cy, xi, alpha;

  public:
    DoubleSphereCamera(const std::array<ftype, 6>& doubleSphereParameters, cv::Size sze = cv::Size(0, 0));
    DoubleSphereCamera(const cv::String& cameraConfigFile);

    // Geometry functions
    std::array<ftype, 6> parameters() const;

    Eigen::Vector3T undistortPoint(const cv::Point2f& point) const override;
    cv::Point2f projectPoint(const Eigen::Vector3T& point) const override;
};

} // namespace GIFT

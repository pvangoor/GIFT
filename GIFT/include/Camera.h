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
#include "opencv2/core/eigen.hpp"
#include <vector>

namespace GIFT {
class Camera {
  protected:
    ftype fx, fy, cx, cy; // intrinsic parameters
    std::vector<ftype> dist;
    std::vector<ftype> invDist;
    std::vector<ftype> computeInverseDistortion() const;

  public:
    cv::Size imageSize;

    Camera(const cv::String& cameraConfigFile);
    Camera(
        cv::Size sze = cv::Size(0, 0), cv::Mat K = cv::Mat::eye(3, 3, CV_64F), std::vector<ftype> dist = {0, 0, 0, 0});

    // Geometry functions
    cv::Mat K() const; // intrinsic matrix (3x3)
    const std::vector<ftype>& distortion() const;

    cv::Point2f undistortPoint(const cv::Point2f& point) const;
    static cv::Point2f distortNormalisedPoint(const cv::Point2f& normalPoint, const std::vector<ftype>& dist);
};

} // namespace GIFT

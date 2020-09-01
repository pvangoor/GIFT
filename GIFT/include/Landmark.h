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

#include "ftype.h"
#include <vector>
#include <array>
#include "eigen3/Eigen/Dense"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

using colorVec = std::array<uchar, 3>;

namespace GIFT {

struct Landmark {
    cv::Point2f camCoordinates;
    cv::Point2f camCoordinatesNorm;

    Eigen::Vector2T opticalFlowRaw;
    Eigen::Vector2T opticalFlowNorm;

    cv::KeyPoint keypoint;

    colorVec pointColor;    
    int idNumber;
    int lifetime = 0;

    Landmark() {};
    Landmark(const cv::Point2f& newCamCoords, const cv::Point2f& newCamCoordsNorm, int idNumber, const colorVec& col = {0,0,0});
    void update(const cv::Point2f& newCamCoords, const cv::Point2f& newCamCoordsNorm, const colorVec& col = {0,0,0});
    Eigen::Vector3T sphereCoordinates() const;
    Eigen::Vector3T opticalFlowSphere() const;
};

}

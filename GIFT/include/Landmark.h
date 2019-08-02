#pragma once

#include <vector>
#include <array>
#include "eigen3/Eigen/Dense"
#include "opencv2/core/core.hpp"

using color = std::array<uchar, 3>;

namespace GIFT {

struct Landmark {
    std::vector<cv::Point2f> camCoordinates;
    std::vector<cv::Point2f> camCoordinatesNorm;
    std::vector<Eigen::Vector2d> opticalFlowRaw;
    std::vector<Eigen::Vector2d> opticalFlowNorm;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    std::vector<color> pointColor;
    int idNumber;
    int lifetime = 0;

};

}
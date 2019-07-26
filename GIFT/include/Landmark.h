#pragma once

#include <vector>
#include "eigen3/Eigen/Dense"
#include "opencv2/core/core.hpp"

namespace GIFT {

struct Landmark {
    std::vector<cv::Point2f> camCoordinates;
    std::vector<cv::Point2f> camCoordinatesNorm;
    std::vector<Eigen::Vector2d> opticalFlowRaw;
    std::vector<Eigen::Vector2d> opticalFlowNorm;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    int idNumber;
    int lifetime = 0;
};

}
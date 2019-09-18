#pragma once

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

    Eigen::Vector3d sphereCoordinates;

    Eigen::Vector2d opticalFlowRaw;
    Eigen::Vector2d opticalFlowNorm;
    Eigen::Vector3d opticalFlowSphere;

    cv::KeyPoint keypoint;

    colorVec pointColor;    
    int idNumber;
    int lifetime = 0;

    Landmark() {};
    Landmark(const cv::Point2f& newCamCoords, const cv::Point2f& newCamCoordsNorm, int idNumber, const colorVec& col = {0,0,0});
    void update(const cv::Point2f& newCamCoords, const cv::Point2f& newCamCoordsNorm, const colorVec& col = {0,0,0});

};

}
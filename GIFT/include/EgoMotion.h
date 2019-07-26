#pragma once

#include "eigen3/Eigen/Dense"

namespace GIFT {

struct EgoMotion {
    Eigen::Vector3d linear = Eigen::Vector3d::Zero();
    Eigen::Vector3d angular = Eigen::Vector3d::Zero();
    int numFlowVectors = 0;
};

}
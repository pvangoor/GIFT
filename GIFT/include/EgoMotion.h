#pragma once

#include "Landmark.h"
#include "eigen3/Eigen/Dense"

namespace GIFT {

struct EgoMotion {
    Eigen::Vector3d linearVelocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d angularVelocity = Eigen::Vector3d::Zero();
    double optimisedResidual = INFINITY;
    int numberOfFeatures = 0;

    EgoMotion(const std::vector<GIFT::Landmark>& landmarks);

    void computeFromOF(const std::vector<GIFT::Landmark>& landmarks, int cameraNumber=0);

    Eigen::Vector3d gaussNewtonStep(const Eigen::Vector3d& V, const std::vector<Eigen::Vector2d>& flow, const std::vector<Eigen::Vector2d> y) const;

};

}
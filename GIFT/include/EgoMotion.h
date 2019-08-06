#pragma once

#include "Landmark.h"
#include "eigen3/Eigen/Dense"

using namespace Eigen;
using namespace std;

namespace GIFT {

class EgoMotion {
public:
    Vector3d linearVelocity;
    Vector3d angularVelocity;
    double optimisedResidual = INFINITY;
    int numberOfFeatures;

    static constexpr double optimisationThreshold = 1e-3;
    static constexpr int maxIterations = 10;

    EgoMotion(const vector<GIFT::Landmark>& landmarks);

private:
    static double optimize(const vector<pair<Vector3d, Vector3d>>& flows, Vector3d& linVel, Vector3d& angVel);
    static void optimizationStep(const vector<pair<Vector3d, Vector3d>>& flows, Vector3d& linVel, Vector3d& angVel);
    static double computeResidual(const vector<pair<Vector3d, Vector3d>>& flows, const Vector3d& linVel, const Vector3d& angVel);
};

}
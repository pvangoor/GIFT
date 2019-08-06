#pragma once

#include "Landmark.h"
#include "eigen3/Eigen/Dense"

using namespace Eigen;
using namespace cv;
using namespace std;

namespace GIFT {

class EgoMotion {
public:
    Vector3d linearVelocity;
    Vector3d angularVelocity;
    double optimisedResidual = INFINITY;
    int numberOfFeatures;

    static constexpr double optimisationThreshold = 1e-7;
    static constexpr int maxIterations = 10;

    
    EgoMotion(const vector<GIFT::Landmark>& landmarks);
    vector<pair<Vector3d, Vector3d>> estimateFlows(const vector<GIFT::Landmark>& landmarks) const;
    vector<pair<Point2f, Vector2d>> estimateFlowsNorm(const vector<GIFT::Landmark>& landmarks) const;

private:
    static Vector3d angularFromLinearVelocity(const vector<pair<Vector3d, Vector3d>>& flows, Vector3d& linVel);
    static double optimize(const vector<pair<Vector3d, Vector3d>>& flows, Vector3d& linVel, Vector3d& angVel);
    static void optimizationStep(const vector<pair<Vector3d, Vector3d>>& flows, Vector3d& linVel, Vector3d& angVel);
    static double computeResidual(const vector<pair<Vector3d, Vector3d>>& flows, const Vector3d& linVel, const Vector3d& angVel);
};

}
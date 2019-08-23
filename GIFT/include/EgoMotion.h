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
    int optimisationSteps;
    int numberOfFeatures;

    static constexpr double optimisationThreshold = 1e-8;
    static constexpr int maxIterations = 30;

    
    EgoMotion(const vector<GIFT::Landmark>& landmarks, const double& dt=1);
    EgoMotion(const vector<GIFT::Landmark>& landmarks, const Vector3d& initLinVel, const double& dt=1);
    EgoMotion(const vector<GIFT::Landmark>& landmarks, const Vector3d& initLinVel, const Vector3d& initAngVel, const double& dt=1);
    EgoMotion(const vector<pair<Vector3d, Vector3d>>& sphereFlows);
    EgoMotion(const vector<pair<Vector3d, Vector3d>>& sphereFlows, const Vector3d& initLinVel);
    EgoMotion(const vector<pair<Vector3d, Vector3d>>& sphereFlows, const Vector3d& initLinVel, const Vector3d& initAngVel);
    vector<pair<Vector3d, Vector3d>> estimateFlows(const vector<GIFT::Landmark>& landmarks) const;
    vector<pair<Point2f, Vector2d>> estimateFlowsNorm(const vector<GIFT::Landmark>& landmarks) const;
    static Vector3d estimateAngularVelocity(const vector<pair<Vector3d, Vector3d>>& sphereFlows, const Vector3d& linVel = Vector3d::Zero());

private:
    static Vector3d angularFromLinearVelocity(const vector<pair<Vector3d, Vector3d>>& flows, Vector3d& linVel);
    static pair<int,double> optimize(const vector<pair<Vector3d, Vector3d>>& flows, Vector3d& linVel, Vector3d& angVel);
    static void optimizationStep(const vector<pair<Vector3d, Vector3d>>& flows, Vector3d& linVel, Vector3d& angVel);
    static double computeResidual(const vector<pair<Vector3d, Vector3d>>& flows, const Vector3d& linVel, const Vector3d& angVel);
    static bool voteForLinVelInversion(const vector<pair<Vector3d, Vector3d>>& flows, const Vector3d& linVel, const Vector3d& angVel);
};

}
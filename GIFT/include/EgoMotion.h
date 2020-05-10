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

#include "Landmark.h"
#include "eigen3/Eigen/Dense"

using namespace Eigen;
using namespace cv;
using namespace std;

namespace GIFT {

class EgoMotion {
public:
    Vector3T linearVelocity;
    Vector3T angularVelocity;
    ftype optimisedResidual = INFINITY;
    int optimisationSteps;
    int numberOfFeatures;

    static constexpr ftype optimisationThreshold = 1e-8;
    static constexpr int maxIterations = 30;

    
    EgoMotion(const vector<GIFT::Landmark>& landmarks, const ftype& dt=1);
    EgoMotion(const vector<GIFT::Landmark>& landmarks, const Vector3T& initLinVel, const ftype& dt=1);
    EgoMotion(const vector<GIFT::Landmark>& landmarks, const Vector3T& initLinVel, const Vector3T& initAngVel, const ftype& dt=1);
    EgoMotion(const vector<pair<Vector3T, Vector3T>>& sphereFlows);
    EgoMotion(const vector<pair<Vector3T, Vector3T>>& sphereFlows, const Vector3T& initLinVel);
    EgoMotion(const vector<pair<Vector3T, Vector3T>>& sphereFlows, const Vector3T& initLinVel, const Vector3T& initAngVel);
    vector<pair<Vector3T, Vector3T>> estimateFlows(const vector<GIFT::Landmark>& landmarks) const;
    vector<pair<Point2f, Vector2T>> estimateFlowsNorm(const vector<GIFT::Landmark>& landmarks) const;
    static Vector3T estimateAngularVelocity(const vector<pair<Vector3T, Vector3T>>& sphereFlows, const Vector3T& linVel = Vector3T::Zero());

private:
    static Vector3T angularFromLinearVelocity(const vector<pair<Vector3T, Vector3T>>& flows, Vector3T& linVel);
    static pair<int,ftype> optimize(const vector<pair<Vector3T, Vector3T>>& flows, Vector3T& linVel, Vector3T& angVel);
    static void optimizationStep(const vector<pair<Vector3T, Vector3T>>& flows, Vector3T& linVel, Vector3T& angVel);
    static ftype computeResidual(const vector<pair<Vector3T, Vector3T>>& flows, const Vector3T& linVel, const Vector3T& angVel);
    static bool voteForLinVelInversion(const vector<pair<Vector3T, Vector3T>>& flows, const Vector3T& linVel, const Vector3T& angVel);
};

}
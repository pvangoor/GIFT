#include "EgoMotion.h"
#include <utility>
#include <iostream>

using namespace std;
using namespace Eigen;
using namespace cv;
using namespace GIFT;

EgoMotion::EgoMotion(const vector<pair<Vector3d, Vector3d>>& sphereFlows) {
    Vector3d linVel(0,0,1);
    Vector3d angVel(0,0,0);

    double residual = optimize(sphereFlows, linVel, angVel);
    
    this->optimisedResidual = residual;
    this->linearVelocity = linVel;
    this->angularVelocity = angVel;
    this->numberOfFeatures = sphereFlows.size();
}

EgoMotion::EgoMotion(const std::vector<Landmark>& landmarks) {
    vector<pair<Vector3d, Vector3d>> sphereFlows;
    for (const auto& lm: landmarks) {
        if (lm.lifetime < 2) continue;
        sphereFlows.emplace_back(make_pair(lm.sphereCoordinates,lm.opticalFlowSphere));
    }

    Vector3d linVel(0,0,1);
    Vector3d angVel(0,0,0);

    double residual = optimize(sphereFlows, linVel, angVel);
    
    this->optimisedResidual = residual;
    this->linearVelocity = linVel;
    this->angularVelocity = angVel;
    this->numberOfFeatures = sphereFlows.size();

}

double EgoMotion::optimize(const vector<pair<Vector3d, Vector3d>>& flows, Vector3d& linVel, Vector3d& angVel) {
    double lastResidual = 1e8;
    double residual = computeResidual(flows, linVel, angVel);

    Vector3d bestLinVel = linVel;
    Vector3d bestAngVel = angVel;
    double bestResidual = INFINITY;

    int iteration = 0;
    while ((abs(lastResidual - residual) > optimisationThreshold) && (iteration < maxIterations)) {
        lastResidual = residual;
        optimizationStep(flows, linVel, angVel);
        residual = computeResidual(flows, linVel, angVel);
        ++iteration;

        cout << "residual: " << residual << endl;

        if (residual < bestResidual) {
            bestResidual = residual;
            bestLinVel = linVel;
            bestAngVel = angVel;
        }
    }

    linVel = bestLinVel;
    angVel = bestAngVel;

    return bestResidual;
}

double EgoMotion::computeResidual(const vector<pair<Vector3d, Vector3d>>& flows, const Vector3d& linVel, const Vector3d& angVel) {
    Vector3d wHat = linVel.normalized();

    double residual = 0;
    int normalisationFactor = 0;
    for (const auto& flow : flows) {
        const Vector3d& phi = flow.second;
        const Vector3d& eta = flow.first;

        double res_i = wHat.dot((phi + angVel.cross(eta)).cross(eta));
        residual += pow(res_i,2);
        ++normalisationFactor;
    }
    normalisationFactor = max(normalisationFactor,1);
    residual = residual / normalisationFactor;

    return residual;
}

void EgoMotion::optimizationStep(const std::vector<pair<Vector3d, Vector3d>>& flows, Vector3d& linVel, Vector3d& angVel) {
    auto Proj3 = [](const Vector3d& vec) { return Matrix3d::Identity() - vec*vec.transpose()/vec.squaredNorm(); };

    Vector3d wHat = linVel.normalized();

    Matrix3d tempHess11 = Matrix3d::Zero();
    Matrix3d tempHess12 = Matrix3d::Zero();
    Matrix3d tempHess22 = Matrix3d::Zero();
    Vector3d tempGrad2 = Vector3d::Zero();

    for (const auto& flow : flows) {
        // Each flow is a pair of spherical bearing eta and perpendicular flow vector phi.
        const Vector3d& phi = flow.second;
        const Vector3d& eta = flow.first;

        Vector3d ZOmega = (phi + angVel.cross(eta)).cross(eta);
        Matrix3d ProjEta = Proj3(eta);

        tempHess11 += ZOmega*ZOmega.transpose();
        tempHess12 += wHat.transpose()*ZOmega*ProjEta + ZOmega*wHat.transpose()*ProjEta;
        tempHess22 += ProjEta*wHat*wHat.transpose()*ProjEta;

        tempGrad2 += wHat.transpose()*ZOmega*ProjEta*wHat;
    }

    Matrix<double, 6,6> hessian;
    Matrix<double, 6,1> gradient;

    Matrix3d ProjWHat = Proj3(wHat);
    hessian.block<3,3>(0,0) = ProjWHat*tempHess11*ProjWHat;
    hessian.block<3,3>(0,3) = -ProjWHat*tempHess12;
    hessian.block<3,3>(3,0) = hessian.block<3,3>(0,3).transpose();
    hessian.block<3,3>(3,3) = tempHess22;

    gradient.block<3,1>(0,0) = ProjWHat*tempHess11*wHat;
    gradient.block<3,1>(3,0) = -tempGrad2;

    // Step with Newton's method
    // Compute the solution to Hess^{-1} * grad
    Matrix<double,6,1> step = hessian.bdcSvd(ComputeFullU | ComputeFullV).solve(gradient);
    wHat += -step.block<3,1>(0,0);

    linVel = linVel.norm() * wHat.normalized();
    angVel += -step.block<3,1>(3,0);
}

vector<pair<Point2f, Vector2d>> EgoMotion::estimateFlowsNorm(const vector<GIFT::Landmark>& landmarks) const {
    vector<pair<Vector3d, Vector3d>> flowsSphere = estimateFlows(landmarks);
    vector<pair<Point2f, Vector2d>> flowsNorm;

    for (const auto& flowSphere: flowsSphere) {
        const Vector3d& eta = flowSphere.first;
        const Vector3d& phi = flowSphere.second;

        double eta3 = eta.z();
        Point2f etaNorm(eta.x() / eta3, eta.y() / eta3);
        Vector2d phiNorm;
        phiNorm << 1/eta3 * (phi.x() - etaNorm.x*phi.z()),
                   1/eta3 * (phi.y() - etaNorm.y*phi.z());
        flowsNorm.emplace_back(make_pair(etaNorm, phiNorm));
    }

    return flowsNorm;
}

vector<pair<Vector3d, Vector3d>> EgoMotion::estimateFlows(const vector<GIFT::Landmark>& landmarks) const {
    auto Proj3 = [](const Vector3d& vec) { return Matrix3d::Identity() - vec*vec.transpose()/vec.squaredNorm(); };

    vector<pair<Vector3d, Vector3d>> estFlows;

    for (const auto& lm: landmarks) {
        const Vector3d& eta = lm.sphereCoordinates;
        const Vector3d etaVel = Proj3(eta) * this->linearVelocity;

        double invDepth = 0;
        if (etaVel.norm() > 0) invDepth = etaVel.dot(this->angularVelocity.cross(eta)) / etaVel.squaredNorm();

        Vector3d flow = -this->angularVelocity.cross(eta) + invDepth * etaVel;
        estFlows.emplace_back(make_pair(eta, flow));
    }

    return estFlows;
}
#include "EgoMotion.h"
#include <utility>

using namespace std;
using namespace Eigen;
using namespace GIFT;

EgoMotion::EgoMotion(const std::vector<Landmark>& landmarks) {
    Vector3d linVel(1,0,0);
    Vector3d angVel(0,0,0);

    vector<pair<Vector3d, Vector3d>> flows;
    this->numberOfFeatures = 0;
    for (const auto& lm: landmarks) {
        if (lm.lifetime < 2) continue;
        flows.emplace_back(make_pair(lm.sphereCoordinates,lm.opticalFlowSphere));
        ++this->numberOfFeatures;
    }

    double residual = optimize(flows, linVel, angVel);
    
    this->optimisedResidual = residual;
    this->linearVelocity = linVel;
    this->angularVelocity = angVel;

}

double EgoMotion::optimize(const vector<pair<Vector3d, Vector3d>>& flows, Vector3d& linVel, Vector3d& angVel) {
    double lastResidual = 1e8;
    double residual = computeResidual(flows, linVel, angVel);

    int iteration = 0;
    while ((abs(lastResidual - residual) > optimisationThreshold) && (iteration < maxIterations)) {
        lastResidual = residual;
        optimizationStep(flows, linVel, angVel);
        residual = computeResidual(flows, linVel, angVel);
        ++iteration;
    }

    return residual;
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
    wHat += step.block<3,1>(0,0);

    linVel = linVel.norm() * wHat.normalized();
    angVel += step.block<3,1>(3,0);
}
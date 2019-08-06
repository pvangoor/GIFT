#include "EgoMotion.h"

using namespace std;
using namespace Eigen;
using namespace GIFT;

EgoMotion::EgoMotion(const std::vector<Landmark>& landmarks) {
    this->computeFromOF(landmarks);
}


void EgoMotion::computeFromOF(const std::vector<GIFT::Landmark>& landmarks, int cameraNumber) {
    vector<Vector2d> y, flow;
    for (const auto & landmark: landmarks) {
        Vector2d yi;
        yi << landmark.camCoordinatesNorm.x, landmark.camCoordinatesNorm.y;
        y.emplace_back(yi);
        flow.emplace_back(landmark.opticalFlowRaw);
    }

    Vector3d V(0,1,1);
    for (int i=0;i<10;++i) {
        V = V + 10*gaussNewtonStep(V, flow, y);
        V.normalize();
    }
    this->linearVelocity = V;

}

Vector3d EgoMotion::gaussNewtonStep(const Vector3d& V, const vector<Vector2d>& flow, const vector<Vector2d> y) const {
    auto Pi2 = [](const Vector2d& vec) {return Matrix2d::Identity() - vec * vec.transpose() / (vec.squaredNorm()); };
    auto Pi3 = [](const Vector3d& vec) {return Matrix3d::Identity() - vec * vec.transpose() / (vec.squaredNorm()); };



    int n = flow.size();

    MatrixXd M(2*n,3), Y(2*n,1);
    vector<Matrix<double,2,3>> A(n), AB(n);
    vector<Vector2d> e(n);

    for (int i = 0; i < n; ++i) {
        double u = y[i].x();
        double v = y[i].y();
        A[i] << 1, 0, -u,
                0, 1, -v;
        AB[i] <<   u*v, -(1+u*u),  v,
                 1+v*v,     -u*v, -u;

        e[i] = (A[i]*V).normalized();

        M.block<2,3>(2*i,0) = Pi2(e[i])*AB[i];
        Y.block<2,1>(2*i,0) = Pi2(e[i])*flow[i];
    }

    Matrix3d MTMInv = (M.transpose()*M).inverse();
    MatrixXd MPseudoInv = MTMInv*M.transpose();
    Vector3d omega = MTMInv*M.transpose()*Y;
    MatrixXd residualVec = (Y - M*omega);
    double residual = residualVec.squaredNorm();


    // Compute the Jacobian
    MatrixXd JacY(2*n,3);
    for (int i=0; i<n; ++i) {
        JacY.block<2,3>(2*i,0) = -(e[i].transpose()*flow[i]*Matrix2d::Identity() + e[i]*flow[i].transpose()) * Pi2(e[i]) * A[i] / (A[i]*V).norm();
    }

    MatrixXd JacMOmega1(2*n,3);
    Matrix3d tempJacMOmega2 = Matrix3d::Zero();
    Matrix3d tempJacMOmega3 = Matrix3d::Zero();

    for (int i=0; i<n; ++i) {
        Matrix<double,2,3> temp = (e[i].transpose()*AB[i]*omega*Matrix2d::Identity() + e[i]*(AB[i]*omega).transpose()) * Pi2(e[i]) * A[i] / (A[i]*V).norm();
        JacMOmega1.block<2,3>(2*i,0) = -temp;
        tempJacMOmega2 += -AB[i].transpose() * (e[i].transpose()*flow[i]*Matrix2d::Identity() + e[i]*flow[i].transpose()) * Pi2(e[i]) * A[i] / (A[i]*V).norm();
        tempJacMOmega3 += -AB[i].transpose() * temp;
    }

    MatrixXd Jacobian = JacMOmega1 + MPseudoInv.transpose()*(tempJacMOmega2+tempJacMOmega3);

    Vector3d GNStep = -(Jacobian.transpose()*Jacobian).inverse() * Jacobian.transpose() * residualVec;
    return Pi3(V) * GNStep;
}
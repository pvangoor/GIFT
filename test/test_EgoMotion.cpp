#include "gtest/gtest.h"
#include "EgoMotion.h"

using namespace Eigen;
using namespace std;

class EgoMotionTest : public ::testing::Test {
protected:
    EgoMotionTest(int pointCount = 100) {
        this->pointCount = pointCount;
        for (int i=0; i<pointCount; ++i) {
            Vector3d point = Vector3d::Random()*100; // Points uniform in 100 size cube 
            while (point.norm() < 1e-1) point = Vector3d::Random()*100;
            double rho = 1/point.norm();
            Vector3d eta = point * rho;
            this->bearingsAndInvDepths.emplace_back(make_pair(eta, rho));
        }
    }

    vector<pair<Vector3d, double>> bearingsAndInvDepths;
    int pointCount;
};

TEST_F(EgoMotionTest, ConvergesLinVel) {
    int testCount = 20;
    for (int i = 0; i < testCount; ++i) {
        Vector3d trueLinVel = (Vector3d::Random()).normalized();
        Vector3d trueAngVel = Vector3d::Random();

        vector<pair<Vector3d, Vector3d>> sphereFlows;
        for (const auto& etaRho: bearingsAndInvDepths) {
            Vector3d phi = etaRho.second * (Matrix3d::Identity() - etaRho.first*etaRho.first.transpose()) * trueLinVel - trueAngVel.cross(etaRho.first);
            sphereFlows.emplace_back(make_pair(etaRho.first, phi));
        }

        Vector3d initialLinVel = (trueLinVel + i*trueLinVel.norm()*Vector3d::Random()/testCount).normalized();

        GIFT::EgoMotion egoMotion(sphereFlows, initialLinVel, trueAngVel);
        const Vector3d& estLinVel = egoMotion.linearVelocity.normalized();
        const Vector3d& estAngVel = egoMotion.angularVelocity;

        EXPECT_LE((estAngVel - trueAngVel).norm(), 1e-2);
        EXPECT_LE(pow(estLinVel.dot(trueLinVel),2) - 1, 1e-2);
    }
}

TEST_F(EgoMotionTest, ConvergesAngVel) {
    int testCount = 20;
    for (int i = 0; i < testCount; ++i) {
        Vector3d trueLinVel = (Vector3d::Random()).normalized();
        Vector3d trueAngVel = Vector3d::Random()*4;

        vector<pair<Vector3d, Vector3d>> sphereFlows;
        for (const auto& etaRho: bearingsAndInvDepths) {
            Vector3d phi = etaRho.second * (Matrix3d::Identity() - etaRho.first*etaRho.first.transpose()) * trueLinVel - trueAngVel.cross(etaRho.first);
            sphereFlows.emplace_back(make_pair(etaRho.first, phi));
        }

        Vector3d initialAngVel = trueAngVel + trueAngVel.norm()*i*Vector3d::Random()/testCount;

        GIFT::EgoMotion egoMotion(sphereFlows, trueLinVel, initialAngVel);
        const Vector3d& estLinVel = egoMotion.linearVelocity.normalized();
        const Vector3d& estAngVel = egoMotion.angularVelocity;

        EXPECT_LE((estAngVel - trueAngVel).norm(), 1e-2);
        EXPECT_LE(pow(estLinVel.dot(trueLinVel),2) - 1, 1e-2);
    }
}

TEST_F(EgoMotionTest, ConvergesFullVel) {
    int testCount = 20;
    for (int i = 0; i < testCount; ++i) {
        Vector3d trueLinVel = (Vector3d::Random()).normalized();
        Vector3d trueAngVel = Vector3d::Random()*4;

        vector<pair<Vector3d, Vector3d>> sphereFlows;
        for (const auto& etaRho: bearingsAndInvDepths) {
            Vector3d phi = etaRho.second * (Matrix3d::Identity() - etaRho.first*etaRho.first.transpose()) * trueLinVel - trueAngVel.cross(etaRho.first);
            sphereFlows.emplace_back(make_pair(etaRho.first, phi));
        }

        Vector3d initialLinVel = (trueLinVel + i*trueLinVel.norm()*Vector3d::Random()/testCount).normalized();
        Vector3d initialAngVel = trueAngVel + trueAngVel.norm()*i*Vector3d::Random()/testCount;

        GIFT::EgoMotion egoMotion(sphereFlows, initialLinVel, initialAngVel);
        const Vector3d& estLinVel = egoMotion.linearVelocity.normalized();
        const Vector3d& estAngVel = egoMotion.angularVelocity;

        EXPECT_LE((estAngVel - trueAngVel).norm(), 1e-2);
        EXPECT_LE(pow(estLinVel.dot(trueLinVel),2) - 1, 1e-2);
    }
}

TEST_F(EgoMotionTest, ConvergesAngVelFromLinVel) {
    int testCount = 20;
    for (int i = 0; i < testCount; ++i) {
        Vector3d trueLinVel = (Vector3d::Random()).normalized();
        Vector3d trueAngVel = Vector3d::Random()*4;

        vector<pair<Vector3d, Vector3d>> sphereFlows;
        for (const auto& etaRho: bearingsAndInvDepths) {
            Vector3d phi = etaRho.second * (Matrix3d::Identity() - etaRho.first*etaRho.first.transpose()) * trueLinVel - trueAngVel.cross(etaRho.first);
            sphereFlows.emplace_back(make_pair(etaRho.first, phi));
        }

        Vector3d initialLinVel = (trueLinVel + i*0.25*trueLinVel.norm()*Vector3d::Random()/testCount).normalized();

        GIFT::EgoMotion egoMotion(sphereFlows, initialLinVel);
        const Vector3d& estLinVel = egoMotion.linearVelocity.normalized();
        const Vector3d& estAngVel = egoMotion.angularVelocity;

        EXPECT_LE((estAngVel - trueAngVel).norm(), 1e-2);
        EXPECT_LE(pow(estLinVel.dot(trueLinVel),2) - 1, 1e-2);
    }
}
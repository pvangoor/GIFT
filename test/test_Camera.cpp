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

#include "Camera.h"
#include "gtest/gtest.h"

using namespace Eigen;
using namespace std;
using namespace cv;
using namespace GIFT;

static double norm(const Point2f& p) { return pow(p.x * p.x + p.y * p.y, 0.5); }

TEST(CameraTest, PinholeProject) {
    const Size imageSize = Size(752, 480);
    const double fx = 458.654;
    const double fy = 457.296;
    const double cx = 367.215;
    const double cy = 248.375;
    const Mat K = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    PinholeCamera cam = PinholeCamera(imageSize, K);

    // Test on a grid of points
    constexpr int skip = 30;
    for (int x = 0; x < imageSize.width; x += skip) {
        for (int y = 0; y < imageSize.width; y += skip) {
            const Point2f imagePoint(x, y);
            const Point2f normalPoint((x - cx) / fx, (y - cy) / fy);
            const Point2f estNormalPoint = cam.undistortPointCV(imagePoint);

            const double error = norm(estNormalPoint - normalPoint);
            EXPECT_LE(error, 1e-4);
        }
    }
}

TEST(CameraTest, DoubleSphereReprojection) {
    const Size imageSize = Size(752, 480);
    const double fx = 458.654;
    const double fy = 457.296;
    const double cx = 367.215;
    const double cy = 248.375;
    const double xi = -0.2;
    const double alpha = 0.5;

    DoubleSphereCamera cam{std::array<ftype, 6>{fx, fy, cx, cy, xi, alpha}};

    // Test on a grid of points
    constexpr int skip = 30;
    for (int x = 0; x < imageSize.width; x += skip) {
        for (int y = 0; y < imageSize.width; y += skip) {
            const Point2f imagePoint(x, y);

            const Eigen::Vector3T spherePoint = cam.undistortPoint(imagePoint);
            const Point2f estImagePoint = cam.projectPoint(spherePoint);

            const double error = norm(estImagePoint - imagePoint);
            EXPECT_LE(error, 1e-3);
        }
    }
}
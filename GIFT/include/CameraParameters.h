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

#include "eigen3/Eigen/Dense"
#include "ftype.h"
#include "opencv2/core/core.hpp"
#include "opencv2/core/eigen.hpp"
#include <vector>

namespace GIFT {
struct CameraParameters {
    cv::Size imageSize;
    cv::Mat K; // intrinsic matrix (3x3)
    std::vector<ftype> distortion;
    std::vector<ftype> inverseDistortion;

    CameraParameters(
        cv::Size sze = cv::Size(0, 0), cv::Mat K = cv::Mat::eye(3, 3, CV_64F), std::vector<ftype> dist = {0, 0, 0, 0}) {

        this->imageSize = sze;
        assert(K.rows == 3 && K.cols == 3);
        this->K = K;
        this->distortion = dist;
        this->inverseDistortion = computeInverseDistortion();
    };

    CameraParameters(const cv::String& cameraConfigFile) {

        cv::FileStorage fs(cameraConfigFile, cv::FileStorage::READ);

        std::vector<int> tempSize = {0, 0};
        if (!fs["image_size"].empty()) {
            fs["image_size"] >> tempSize;
        } else if (!fs["size"].empty()) {
            fs["size"] >> tempSize;
        }
        this->imageSize = cv::Size(tempSize[1], tempSize[0]);

        if (!fs["camera_matrix"].empty()) {
            fs["camera_matrix"] >> this->K;
        } else if (!fs["camera"].empty()) {
            fs["camera"] >> this->K;
        } else if (!fs["K"].empty()) {
            fs["K"] >> this->K;
        }

        if (!fs["distortion_coefficients"].empty()) {
            fs["distortion_coefficients"] >> this->distortion;
        } else if (!fs["distortion"].empty()) {
            fs["distortion"] >> this->distortion;
        } else if (!fs["dist"].empty()) {
            fs["dist"] >> this->distortion;
        }

        this->inverseDistortion = computeInverseDistortion();
    };

    cv::Point2f undistortPoint(const cv::Point2f& point) const {
        return cv::Point2f(0, 0); // TODO
    }

    cv::Point2f distortNormalisedPoint(const cv::Point2f& normalPoint) const {
        cv::Point2f distortedPoint = cv::Point2f(0, 0);
        const ftype r2 = normalPoint.x * normalPoint.x + normalPoint.y * normalPoint.y;
        if (distortion.size() >= 2) {
            distortedPoint.x += normalPoint.x * (1 + distortion[0] * r2 + distortion[1] * r2 * r2);
            distortedPoint.y += normalPoint.y * (1 + distortion[0] * r2 + distortion[1] * r2 * r2);
        }
        if (distortion.size() >= 4) {
            distortedPoint.x += 2 * distortion[2] * normalPoint.x * normalPoint.y +
                                distortion[3] * (r2 + 2 * normalPoint.x * normalPoint.x);
            distortedPoint.y += 2 * distortion[3] * normalPoint.x * normalPoint.y +
                                distortion[2] * (r2 + 2 * normalPoint.y * normalPoint.y);
        }
        if (distortion.size() >= 5) {
            distortedPoint.x += normalPoint.x * distortion[4] * r2 * r2 * r2;
            distortedPoint.y += normalPoint.y * distortion[4] * r2 * r2 * r2;
        }
        return distortedPoint;
    }

    std::vector<ftype> computeInverseDistortion() const {
        const ftype cx = ftype(K.at<double>(0, 2));
        const ftype cy = ftype(K.at<double>(1, 2));
        const ftype fx = ftype(K.at<double>(0, 0));
        const ftype fy = ftype(K.at<double>(1, 1));
        const cv::Size compSize = imageSize.area() == 0 ? cv::Size(int(round(cx * 2)), int(round(cy * 2))) : imageSize;

        // Construct a vector of normalised points
        constexpr int maxPoints = 30;
        std::vector<cv::Point2f> distortedPoints;
        std::vector<cv::Point2f> normalPoints;
        for (int x = 0; x < imageSize.width; x += imageSize.width / maxPoints) {
            for (int y = 0; y < imageSize.height; x += imageSize.height / maxPoints) {
                // Normalise and distort the point
                const cv::Point2f normalPoint((x - cx) / fx, (y - cy) / fy);
                const cv::Point2f distPoint = distortNormalisedPoint(normalPoint);

                normalPoints.emplace_back(normalPoint);
                distortedPoints.emplace_back(distPoint);
            }
        }

        // Set up the least squares problem
        Eigen::Matrix<ftype, Eigen::Dynamic, 5> lmat(distortedPoints.size() * 2, 5);
        Eigen::Matrix<ftype, Eigen::Dynamic, 1> rvec(distortedPoints.size() * 2, 1);
        for (int i = 0; i < distortedPoints.size(); ++i) {
            const cv::Point2f& p = distortedPoints[i];
            const ftype r2 = p.x * p.x + p.y * p.y;
            lmat.block<2, 5>(2 * i, 0) << p.x * r2, p.x * r2 * r2, 2 * p.x * p.y, r2 + 2 * p.x * p.x,
                p.x * r2 * r2 * r2, p.y * r2, p.y * r2 * r2, r2 + 2 * p.y * p.y, 2 * p.x * p.y, p.y * r2 * r2 * r2;
            rvec.block<2, 1>(2 * i, 0) << normalPoints[i].x, normalPoints[i].y;
        }
        const Eigen::Matrix<ftype, 5, 1> invDist = lmat.ldlt().solve(rvec);
        std::vector<ftype> invDistVec(invDist.data(), invDist.data() + invDist.rows());
        return invDistVec;
    }
};
} // namespace GIFT

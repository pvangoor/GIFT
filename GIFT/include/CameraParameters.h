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

#include "ftype.h"
#include "eigen3/Eigen/Dense"
#include "opencv2/core/core.hpp"
#include "opencv2/core/eigen.hpp"
#include <vector>

namespace GIFT {
struct CameraParameters {
    Eigen::Matrix4T pose;
    cv::Mat K; // intrinsic matrix (3x3)
    std::vector<ftype> distortionParams;
    Eigen::Matrix<ftype, 3,4> P; // Rectified projection matrix

    CameraParameters(cv::Mat K = cv::Mat::eye(3,3,CV_64F), Eigen::Matrix4T pose=Eigen::Matrix4T::Identity(), std::vector<ftype> distortionParams={0,0,0,0}) {
        assert(K.rows == 3 && K.cols == 3);
        
        this->K = K;
        this->distortionParams = distortionParams;
        this->pose = pose;
        
        this->P.setZero();
        Eigen::Matrix3T eigenK;
        cv::cv2eigen(K, eigenK);
        this->P.block<3,3>(0,0) = eigenK;
        this->P = this->P * this->pose.inverse();
    };

    CameraParameters(const cv::String& cameraConfigFile) {

        cv::FileStorage fs(cameraConfigFile, cv::FileStorage::READ);
        fs["camera_matrix"] >> this->K;

        cv::Mat dist;
        fs["distortion_coefficients"] >> dist;
        this->distortionParams = dist;

        this->pose = Eigen::Matrix4T::Identity();
        
        this->P.setZero();
        Eigen::Matrix3T eigenK;
        cv::cv2eigen(K, eigenK);
        this->P.block<3,3>(0,0) = eigenK;
        this->P = this->P * this->pose.inverse();
    };
};
}

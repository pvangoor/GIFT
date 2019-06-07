#include "eigen3/Eigen/Dense"
#include "opencv2/core/core.hpp"
#include "opencv2/core/eigen.hpp"
#include <vector>

struct CameraParameters {
    CameraParameters(cv::Mat K, std::vector<double> distortionParams={0,0,0,0}, Eigen::Matrix4d pose=Eigen::Matrix4d::Identity()) {
        assert(K.rows == 3 && K.cols == 3);
        
        this->K = K;
        this->distortionParams = distortionParams;
        this->pose = pose;
        
        this->P.setZero();
        Eigen::Matrix3d eigenK;
        cv::cv2eigen(K, eigenK);
        this->P.block<3,3>(0,0) = eigenK;
        this->P = this->P * this->pose.inverse();
    };
    Eigen::Matrix4d pose;
    cv::Mat K; // intrinsic matrix (3x3)
    std::vector<double> distortionParams;
    Eigen::Matrix<double, 3,4> P; // Rectified projection matrix
};
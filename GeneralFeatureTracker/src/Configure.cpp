#include "Configure.h"
#include "vector"

// using namespace GFT;

GFT::CameraParameters GFT::readCameraConfig(const std::string &fileName) {
    YAML::Node config = YAML::LoadFile(fileName);

    Eigen::Matrix3d K;
    if (config["K"]) {
        K = GFT::convertYamlToMatrix(config["K"]);
    } else {
        throw "The intrinsic matrix is not given.";
    }
    cv::Mat cvK;
    cv::eigen2cv(K, cvK);
     
    Eigen::Matrix4d pose;
    if (config["pose"]) {
        pose = GFT::convertYamlToMatrix(config["pose"]);
    } else {
        pose.setIdentity();
    }

    GFT::CameraParameters cam = GFT::CameraParameters(cvK, pose);

    if (config["distortionParams"]) {
        std::vector<double> distortionParams;
        for (int i=0; i<config["distortionParams"].size(); ++i) {
            distortionParams.emplace_back(config["distortionParams"][i].as<double>());
        }
        cam.distortionParams = distortionParams;
    }

    return cam;
}

Eigen::MatrixXd GFT::convertYamlToMatrix(YAML::Node yaml) {
    int m = yaml["rows"].as<int>();
    int n = yaml["cols"].as<int>();
    Eigen::MatrixXd mat(m, n);
    int k = 0;
    for (int i=0;i<m;++i) {
        for (int j=0;j<n;++j) {
            mat(i,j) = yaml["data"][k].as<double>();
            ++k;
        }
    }
    return mat;
}
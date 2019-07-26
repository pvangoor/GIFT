#include "Configure.h"
#include "vector"

using namespace GIFT;

CameraParameters GIFT::readCameraConfig(const std::string &fileName) {
    YAML::Node config = YAML::LoadFile(fileName);

    Eigen::Matrix3d K;
    if (config["K"]) {
        K = convertYamlToMatrix(config["K"]);
    } else {
        throw "The intrinsic matrix is not given.";
    }
    cv::Mat cvK;
    cv::eigen2cv(K, cvK);
     
    Eigen::Matrix4d pose;
    if (config["pose"]) {
        pose = convertYamlToMatrix(config["pose"]);
    } else {
        pose.setIdentity();
    }

    CameraParameters cam = CameraParameters(cvK, pose);

    if (config["distortionParams"]) {
        std::vector<double> distortionParams;
        for (int i=0; i<config["distortionParams"].size(); ++i) {
            distortionParams.emplace_back(config["distortionParams"][i].as<double>());
        }
        cam.distortionParams = distortionParams;
    }

    return cam;
}

Eigen::MatrixXd GIFT::convertYamlToMatrix(YAML::Node yaml) {
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
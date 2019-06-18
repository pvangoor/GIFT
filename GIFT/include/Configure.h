#pragma once

#include "CameraParameters.h"
#include "string"
#include "yaml-cpp/yaml.h"

namespace GFT {

GFT::CameraParameters readCameraConfig(const std::string &fileName);

// int testFunc() {
//     return 5;
// }

Eigen::MatrixXd convertYamlToMatrix(YAML::Node yaml);


}
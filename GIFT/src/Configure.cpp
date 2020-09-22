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

#include "Configure.h"
#include "ftype.h"
#include "vector"
#include <stdexcept>

using namespace GIFT;

Camera GIFT::readCameraConfig(const std::string& fileName) {
    YAML::Node config = YAML::LoadFile(fileName);

    Eigen::Matrix3T K;
    if (config["K"]) {
        K = convertYamlToMatrix(config["K"]);
    } else if (config["camera_matrix"]) {
        K = convertYamlToMatrix(config["camera_matrix"]);
    } else {
        throw std::invalid_argument("The intrinsic matrix is not given.");
    }
    cv::Mat cvK;
    cv::eigen2cv(K, cvK);

    std::vector<ftype> distortionParams;
    if (config["distortionParams"]) {
        for (int i = 0; i < config["distortionParams"].size(); ++i) {
            distortionParams.emplace_back(config["distortionParams"][i].as<ftype>());
        }
    }

    Camera cam = Camera(cv::Size(0, 0), cvK, distortionParams);

    return cam;
}

Eigen::MatrixXT GIFT::convertYamlToMatrix(YAML::Node yaml) {
    int m = yaml["rows"].as<int>();
    int n = yaml["cols"].as<int>();
    Eigen::MatrixXT mat(m, n);
    int k = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            mat(i, j) = yaml["data"][k].as<ftype>();
            ++k;
        }
    }
    return mat;
}

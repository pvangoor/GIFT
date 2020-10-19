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
#include "PointFeatureTracker.h"

#include "opencv2/highgui/highgui.hpp"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char* argv[]) {

    if (argc != 3 && argc != 4) {
        std::cout << "Usage: VisualOdometryTracking camera.yaml video.mp4 (settings.yaml)" << std::endl;
        exit(1);
    }

    // Set up the feature tracker
    GIFT::Camera camera = GIFT::Camera(cv::String(argv[1]));
    GIFT::PointFeatureTracker ft = GIFT::PointFeatureTracker(camera);
    ft.maxFeatures = 30;
    ft.featureDist = 30;
    ft.minHarrisQuality = 0.05;
    ft.featureSearchThreshold = 0.8;

    if (argc == 4) {
        // std::cout << "Reading filter settings from " << argv[3] << std::endl;
        const YAML::Node configNode = YAML::LoadFile(std::string(argv[3]));
        ft.maxFeatures = configNode["maxFeatures"].as<int>();
        ft.featureDist = configNode["featureDist"].as<ftype>();
        ft.minHarrisQuality = configNode["minHarrisQuality"].as<ftype>();
        ft.featureSearchThreshold = configNode["featureSearchThreshold"].as<ftype>();
        ft.maxError = configNode["maxError"].as<float>();
        ft.winSize = configNode["winSize"].as<int>();
    }

    // Set up the video capture
    cv::VideoCapture cap;
    cap.open(cv::String(argv[2]));
    cv::Mat image;

    // Set up the output file
    std::time_t t = std::time(nullptr);
    std::stringstream outputFileNameStream, internalFileNameStream;
    outputFileNameStream << "GIFT_Monocular_" << std::put_time(std::localtime(&t), "%F_%T") << ".csv";
    std::ofstream outputFile(outputFileNameStream.str());
    outputFile << "frame, N, eta1id, eta1x, eta1y, eta1z, ..., ..., ..., ..., etaNid, etaNx, etaNy, etaNz" << std::endl;

    int frameCounter = 0;
    while (cap.read(image)) {

        ft.processImage(image);
        std::vector<GIFT::Feature> features = ft.outputLandmarks();

        // Write features to file
        outputFile << frameCounter << ", ";
        outputFile << features.size();
        for (const GIFT::Feature f : features) {
            outputFile << ", " << f.idNumber;
            outputFile << ", " << f.sphereCoordinates().format(IOFormat(-1, 0, ", ", ", "));
        }
        outputFile << std::endl;

        ++frameCounter;
    }
}
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

#include "Camera.h"
#include "EgoMotion.h"
#include "Feature.h"
#include "eigen3/Eigen/Dense"
#include "ftype.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <memory>
#include <vector>

namespace GIFT {

Eigen::Matrix3T skew_matrix(const Eigen::Vector3T& t);

class PointFeatureTracker {
  protected:
    std::shared_ptr<Camera> cameraPtr;

    // Variables used in the tracking algorithms
    int currentNumber = 0;
    cv::Mat previousImage;
    std::vector<Feature> features;
    cv::Mat imageMask;

  public:
    int maxFeatures = 500;
    ftype featureDist = 20;
    ftype minHarrisQuality = 0.1;
    ftype featureSearchThreshold = 1.0;
    float maxError = 1e8;
    int winSize = 21;
    int maxLevel = 3;

    // // Stereo Specific
    // ftype stereoBaseline = 0.1;
    // ftype stereoThreshold = 1;

  public:
    // Initialisation and configuration
    PointFeatureTracker(const Camera& configuration = Camera()) {
        cameraPtr = std::make_shared<Camera>(configuration);
    };

    void setCameraConfiguration(const Camera& configuration) { cameraPtr = std::make_shared<Camera>(configuration); }

    // Core
    void processImage(const cv::Mat& image);
    std::vector<Feature> outputFeatures() const { return features; };

    // Visualisation
    cv::Mat drawFeatureImage(
        const cv::Scalar& color = cv::Scalar(0, 0, 255), const int pointSize = 2, const int thickness = 1) const;
    cv::Mat drawFlowImage(const cv::Scalar& featureColor = cv::Scalar(0, 0, 255),
        const cv::Scalar& flowColor = cv::Scalar(0, 255, 255), const int pointSize = 2, const int thickness = 1) const;
    cv::Mat drawFlow(const cv::Scalar& featureColor = cv::Scalar(0, 0, 255),
        const cv::Scalar& flowColor = cv::Scalar(0, 255, 255), const int pointSize = 2, const int thickness = 1) const;

    // Masking
    void setMask(const cv::Mat& mask, int cameraNumber = 0);

    // EgoMotion
    EgoMotion computeEgoMotion(int minLifetime = 1) const;

  protected:
    std::vector<cv::Point2f> detectNewFeatures(const cv::Mat& image) const;
    std::vector<cv::Point2f> removeDuplicateFeatures(const std::vector<cv::Point2f>& proposedFeatures) const;
    std::vector<Feature> createNewFeatures(const cv::Mat& image, const std::vector<cv::Point2f>& newFeatures);

    void trackFeatures(const cv::Mat& image);
    void addNewFeatures(std::vector<Feature> newFeatures);
    void computeLandmarkPositions();
};

} // namespace GIFT

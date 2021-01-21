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

#include "EgoMotion.h"

#include "GIFeatureTracker.h"

#include "eigen3/Eigen/Dense"

#include <memory>
#include <vector>

namespace GIFT {

Eigen::Matrix3T skew_matrix(const Eigen::Vector3T& t);

class PointFeatureTracker : public GIFeatureTracker {
  protected:
    cv::Mat previousImage;
    std::vector<Feature> features;

  public:
    ftype featureDist = 20;
    ftype minHarrisQuality = 0.1;
    float maxError = 1e8;
    int winSize = 21;
    int maxLevel = 3;

  public:
    // Initialisation and configuration
    PointFeatureTracker() : GIFeatureTracker() {}
    PointFeatureTracker(const Camera& cameraParams) : GIFeatureTracker(cameraParams) {}
    PointFeatureTracker(const Camera& cameraParams, const cv::Mat& mask) : GIFeatureTracker(cameraParams, mask) {}

    // Core
    virtual void processImage(const cv::Mat& image) override;
    virtual void detectFeatures(const cv::Mat& image) override;
    virtual void trackFeatures(const cv::Mat& image) override;
    virtual void useFeaturePredictions(const std::vector<Feature>& predictedFeatures);
    std::vector<Feature> outputFeatures() const { return features; };

    // EgoMotion
    EgoMotion computeEgoMotion(int minLifetime = 1) const;

  protected:
    std::vector<cv::Point2f> identifyFeatureCandidates(const cv::Mat& image) const;
    std::vector<cv::Point2f> removeDuplicateFeatures(const std::vector<cv::Point2f>& proposedFeatures) const;
    std::vector<Feature> createNewFeatures(const cv::Mat& image, const std::vector<cv::Point2f>& newFeatures);

    void addNewFeatures(std::vector<Feature> newFeatures);
    void computeLandmarkPositions();
};

} // namespace GIFT

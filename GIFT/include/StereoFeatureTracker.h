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

#include "PointFeatureTracker.h"
#include "StereoLandmark.h"

namespace GIFT {

enum class StereoCam { Left, Right };

class StereoFeatureTracker {

  protected:
    PointFeatureTracker trackerLeft;
    PointFeatureTracker trackerRight;
    std::vector<StereoLandmark> stereoLandmarks;

  public:
    // Stereo Specific
    ftype stereoBaseline = 0.1;
    ftype stereoThreshold = 1;

  public:
    // Initialisation
    StereoFeatureTracker(const Camera& camLeft, const Camera& camRight) {
        trackerLeft.setCameraConfiguration(camLeft);
        trackerRight.setCameraConfiguration(camRight);
    };

    // Configuration
    void setCameraConfiguration(const Camera& camLeft, const Camera& camRight) {
        trackerLeft.setCameraConfiguration(camLeft);
        trackerRight.setCameraConfiguration(camRight);
    };
    void setCameraConfiguration(const Camera& configuration, StereoCam stereoCam = StereoCam::Left) {
        if (stereoCam == StereoCam::Left)
            trackerLeft.setCameraConfiguration(configuration);
        else
            trackerRight.setCameraConfiguration(configuration);
    }
    void setMask(const cv::Mat& mask, StereoCam stereoCam = StereoCam::Left) {
        if (stereoCam == StereoCam::Left)
            trackerLeft.setMask(mask);
        else
            trackerRight.setMask(mask);
    }

    // Core
    void processImages(const cv::Mat& imageLeft, const cv::Mat& imageRight);
    std::vector<StereoLandmark> outputStereoLandmarks() const { return stereoLandmarks; };

  protected:
    void removeLostStereoLandmarks(
        const std::vector<Feature>& landmarksLeft, const std::vector<Feature>& landmarksRight);
    std::vector<StereoLandmark> createNewStereoLandmarks(const std::vector<Feature>& landmarksLeft,
        const cv::Mat& imageLeft, const std::vector<Feature>& landmarksRight, const cv::Mat& imageRight) const;
    void addNewStereoLandmarks(const std::vector<StereoLandmark>& newStereoLandmarks);
};

} // namespace GIFT
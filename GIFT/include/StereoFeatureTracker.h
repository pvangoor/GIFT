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

#include "FeatureTracker.h"
#include "StereoLandmark.h"


using namespace Eigen;
using namespace std;
using namespace cv;

namespace GIFT {

enum class StereoCam {Left, Right};

class StereoFeatureTracker {

protected:
    FeatureTracker trackerLeft;
    vector<StereoLandmark> stereoLandmarks;

public:
    // Stereo Specific
    double stereoBaseline = 0.1;
    double stereoThreshold = 1;

public:
    // Initialisation
    StereoFeatureTracker(const CameraParameters &camLeft, const CameraParameters &camRight) {
        trackerLeft.setCameraConfiguration(camLeft);
        trackerRight.setCameraConfiguration(camRight);
    };

    // Configuration
    void setCameraConfiguration(const CameraParameters &camLeft, const CameraParameters &camRight) {
        trackerLeft.setCameraConfiguration(camLeft);
        trackerRight.setCameraConfiguration(camRight);
    };
    void setCameraConfiguration(const CameraParameters &configuration, StereoCam stereoCam = StereoCam::Left) {
        if (stereoCam == StereoCam::Left) trackerLeft.setCameraConfiguration(configuration);
        else trackerRight.setCameraConfiguration(configuration);
    }
    void setMask(const Mat & mask, StereoCam stereoCam = StereoCam::Left) {
        if (stereoCam == StereoCam::Left) trackerLeft.setMask(mask);
        else trackerRight.setMask(mask);
    }

    // Core
    void processImages(const Mat &imageLeft, const Mat &imageRight);
    vector<StereoLandmark> outputStereoLandmarks() const { return stereoLandmarks; };

    // Visualisation
    Mat drawFeatureImage(const Scalar& color = Scalar(0,0,255), const int pointSize = 2, const int thickness = 1) const;

protected:
    void removeLostStereoLandmarks(const vector<Landmark>& landmarksLeft, const vector<Landmark>& landmarksRight);
    vector<StereoLandmark> createNewStereoLandmarks(vector<Landmark>& landmarksLeft, const Mat& imageLeft,
                                                    vector<Landmark>& landmarksRight, const Mat& imageRight) const;
    void addNewStereoLandmarks(const vector<StereoLandmark>& newStereoLandmarks);

    // Stereo Landmark matching
    vector<StereoLandmark> findStereoLandmarks(vector<Landmark>& landmarksLeft, const Mat& imageLeft, const Mat& imageRight,
                                                Size winSize = Size(21,21), int maxLevel = 3) const;
};

}
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

#include "ftype.h"
#include "Landmark.h"
#include "EgoMotion.h"
#include "CameraParameters.h"
#include "eigen3/Eigen/Dense"
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"


using namespace Eigen;
using namespace std;
using namespace cv;

namespace GIFT {

Eigen::Matrix3T skew_matrix(const Eigen::Vector3T& t);

class FeatureTracker {
protected:
    CameraParameters camera;

    // Variables used in the tracking algorithms
    int currentNumber = 0;
    Mat previousImage;
    vector<Landmark> landmarks;
    Mat imageMask;

public:
    int maxFeatures = 500;
    ftype featureDist = 20;
    ftype minHarrisQuality = 0.1;
    ftype featureSearchThreshold = 1.0;

    // // Stereo Specific
    // ftype stereoBaseline = 0.1;
    // ftype stereoThreshold = 1;

public:
    // Initialisation and configuration
    FeatureTracker(const CameraParameters &configuration = CameraParameters()) { camera = configuration; };
    void setCameraConfiguration(const CameraParameters &configuration);

    // Core
    void processImage(const Mat &image);
    vector<Landmark> outputLandmarks() const { return landmarks; };

    // Visualisation
    Mat drawFeatureImage(const Scalar& color = Scalar(0,0,255), const int pointSize = 2, const int thickness = 1) const;
    Mat drawFlowImage(const Scalar& featureColor = Scalar(0,0,255), const Scalar& flowColor = Scalar(0,255,255), const int pointSize = 2, const int thickness = 1) const;
    Mat drawFlow(const Scalar& featureColor = Scalar(0,0,255), const Scalar& flowColor = Scalar(0,255,255), const int pointSize = 2, const int thickness = 1) const;

    // Masking
    void setMask(const Mat & mask, int cameraNumber=0);

    // EgoMotion
    EgoMotion computeEgoMotion(int minLifetime=1) const;

protected:
    vector<Point2f> detectNewFeatures(const Mat &image) const;
    vector<Point2f> removeDuplicateFeatures(const vector<Point2f> &proposedFeatures) const;
    vector<Landmark> createNewLandmarks(const Mat &image, const vector<Point2f>& newFeatures);

    void trackLandmarks(const Mat &image);
    void addNewLandmarks(vector<Landmark> newLandmarks);
    void computeLandmarkPositions();
};

}

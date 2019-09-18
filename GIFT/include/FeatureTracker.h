//
// Created by pieter on 31/03/19.
//

#pragma once

#include "Landmark.h"
#include "CameraParameters.h"
#include "eigen3/Eigen/Dense"
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"


using namespace Eigen;
using namespace std;
using namespace cv;

namespace GIFT {

Eigen::Matrix3d skew_matrix(const Eigen::Vector3d& t);

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
    double featureDist = 20;
    double minHarrisQuality = 0.1;

    // // Stereo Specific
    // double stereoBaseline = 0.1;
    // double stereoThreshold = 1;

public:
    // Initialisation and configuration
    FeatureTracker(const CameraParameters &configuration) { camera = configuration; };
    void setCameraConfiguration(const CameraParameters &configuration);

    // Core
    void processImage(const Mat &image);
    vector<Landmark> outputLandmarks() const { return landmarks; };

    // Visualisation
    Mat drawFeatureImage(const Scalar& color = Scalar(0,0,255), const int pointSize = 2, const int thickness = 1) const;
    Mat drawFlowImage(const Scalar& featureColor = Scalar(0,0,255), const Scalar& flowColor = Scalar(0,255,255), const int pointSize = 2, const int thickness = 1) const;

    // Masking
    void setMask(const Mat & mask, int cameraNumber=0);

protected:
    vector<Point2f> detectNewFeatures(const Mat &image) const;
    vector<Point2f> removeDuplicateFeatures(const vector<Point2f> &proposedFeatures) const;
    vector<Landmark> createNewLandmarks(const Mat &image, const vector<Point2f>& newFeatures);

    void trackLandmarks(const Mat &image);
    void addNewLandmarks(vector<Landmark> newLandmarks);
    void computeLandmarkPositions();
};

}
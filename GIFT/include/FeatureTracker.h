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

enum class TrackerMode {
    MONO, STEREO, MULTIVIEW
};

Eigen::Matrix3d skew_matrix(const Eigen::Vector3d& t);

class FeatureTracker {
protected:
    // Core settings
    TrackerMode mode = TrackerMode::MONO;
    vector<CameraParameters> cameras;

    // Variables used in the tracking algorithms
    int currentNumber = 0;
    vector<Mat> previousImages;
    vector<Landmark> landmarks;
    vector<Mat> imageMasks;

public:
    int maxFeatures = 500;
    double featureDist = 20;

    // Stereo Specific
    double stereoBaseline = 0.1;
    double stereoThreshold = 1;

public:
    // Initialisation and configuration
    FeatureTracker(TrackerMode mode = TrackerMode::MONO);
    void setCameraConfiguration(const CameraParameters &configuration, int cameraNumber=0);

    // Core
    void processImages(const vector<Mat> &images);
    void processImage(const Mat &image);
    vector<Landmark> outputLandmarks() const { return landmarks; };

    // Masking
    void setMask(const Mat & mask, int cameraNumber=0);
    void setMasks(const vector<Mat> & masks);

protected:
    vector<Point2f> detectNewFeatures(const Mat &image, int cameraNumber=0) const;
    vector<vector<Point2f>> detectNewStereoFeatures(const cv::Mat & imageLeft, const cv::Mat &imageRight) const;
    vector<Point2f> removeDuplicateFeatures(const vector<Point2f> &proposedFeatures, int cameraNumber=0) const;
    vector<Landmark> matchImageFeatures(vector<vector<Point2f>> features, vector<vector<Point2f>> featuresNorm, vector<Mat> images) const;

    void trackLandmarks(const Mat &image, int cameraNumber);
    void addNewLandmarks(vector<Landmark> newLandmarks);
    void computeLandmarkPositions();

    Vector3d solveStereo(const Point2f& leftKp, const Point2f& rightKp) const;
    Vector3d solveMultiView(const vector<Point2f> imageCoordinates) const;
    bool checkStereoQuality(const Point2f &leftKp, const Point2f &rightKp) const;
};

}
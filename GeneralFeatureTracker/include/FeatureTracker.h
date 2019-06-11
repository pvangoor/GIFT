//
// Created by pieter on 31/03/19.
//

#pragma once

#include "CameraParameters.h"
#include "eigen3/Eigen/Dense"
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"


using namespace Eigen;
using namespace std;
using namespace cv;

namespace GFT {

namespace MODE {
    enum TrackerMode {
        MONO, STEREO, MULTIVIEW
    };
}

struct Landmark {
    vector<Point2f> camCoordinates;
    Vector3d position;
    int idNumber;
};

Eigen::Matrix3d skew_matrix(const Eigen::Vector3d& t);

class FeatureTracker {
private:

    // Tracker Settings
    int numFeatures = 500;
    double featureDist = 20;
    bool display = false;
    Ptr<Feature2D> describer = ORB::create(numFeatures, 1.2f, 8, 63, 0, 2, ORB::HARRIS_SCORE, 63); // feature describer;
    Ptr<BFMatcher> bfMatcher = BFMatcher::create(cv::NORM_HAMMING, true);

    MODE::TrackerMode mode = MODE::MONO;

    double fundamentalThreshold = 0.01;

    double stereoBaseline = 0.1;
    double stereoThreshold = 1;
    
    double minDepth = 0.0;
    double maxDepth = 1e6;

    Matrix3d fundamentalMatrix;

    vector<CameraParameters> cameras;

    int currentNumber = 0;
    vector<Mat> previousImages;
    vector<Landmark> landmarks;


public:
    FeatureTracker(MODE::TrackerMode mode = MODE::MONO);

    void setCameraConfiguration(int cameraNumber, const CameraParameters &configuration);

    void processImages(const vector<Mat> &images);
    void processImage(const Mat &image);

    vector<Landmark> outputLandmarks() const { return landmarks; };

protected:
    MODE::TrackerMode readMode(String modeName);

    vector<Point2f> detectNewFeatures(const Mat &image, int cameraNumber=0) const;
    vector<Landmark> matchImageFeatures(vector<vector<Point2f>> features, vector<Mat> images) const;
    void addNewLandmarks(vector<Landmark> newLandmarks);
    void computeLandmarkPositions();

    Vector3d solveStereo(const Point2f& leftKp, const Point2f& rightKp) const;
    Vector3d triangulateLS(const Point2f& leftKp, const Point2f& rightKp);
    bool checkStereoQuality(const Point2f &leftKp, const Point2f &rightKp);
    void trackLandmarks(const Mat &image, int cameraNumber);
//    void trackExistingFeatures(const cv::Mat &image, vector<Vector3d>& features)

};

}
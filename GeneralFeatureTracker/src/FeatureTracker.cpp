//
// Created by pieter on 31/03/19.
//

#include <FeatureTracker.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/stereo/stereo.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "eigen3/Eigen/SVD"
#include "iostream"
#include "string"

void FeatureTracker::processImage(const Mat &image) {
    processImages(vector<Mat>({image}));
}

void FeatureTracker::processImages(const vector<Mat> &images) {
    assert(images.size() == cameras.size());

    // Track the existing features
    for (int i=0; i < cameras.size(); ++i) {
        this->trackLandmarks(images[i], i);
    }
    this->previousImages = images;

    // Find potential new features
    vector<vector<Point2f>> newFeatures;
    for (int i=0; i<images.size(); ++i) {
        vector<Point2f> cameraFeatures = this->detectNewFeatures(images[i]);
        cv::undistortPoints(cameraFeatures, cameraFeatures, cameras[i].K, cameras[i].distortionParams);
        newFeatures.emplace_back(cameraFeatures);
    }

    vector<Landmark> newLandmarks = this->matchImageFeatures(newFeatures, images);
    this->addNewLandmarks(newLandmarks);
    this->computeLandmarkPositions();

}

void FeatureTracker::computeLandmarkPositions() {
    for (auto & lm : landmarks) {
        if (mode == MODE::MONO) {
            lm.position << lm.camCoordinates[0].x, lm.camCoordinates[0].y, 1;
            lm.position.normalize();
        }

        if (mode == MODE::STEREO) {
            lm.position = this->solveStereo(lm.camCoordinates[0], lm.camCoordinates[1]);
        }
    }
}

Vector3d FeatureTracker::solveStereo(const Point2f& leftKp, const Point2f& rightKp) const {
    Vector3d position;
    position << leftKp.x, leftKp.y, 1;

    double scale = stereoBaseline / (leftKp.x - rightKp.x);
    position = scale*position;

    return position;
}

// Vector3d FeatureTracker::triangulateLS(const Point2f &leftKp, const Point2f &rightKp) {
//     Matrix<double,6,4> solutionMat;

//     Vector3d eigLeftKp, eigRightKp;
//     eigLeftKp << leftKp.x, leftKp.y, 1;
//     eigRightKp << rightKp.x, rightKp.y, 1;

//     solutionMat.block<3,4>(0,0) = skew_matrix(eigLeftKp) * leftCam.P;
//     solutionMat.block<3,4>(3,0) = skew_matrix(eigRightKp) * rightCam.P;

//     JacobiSVD<MatrixXd> svd(solutionMat, ComputeThinU | ComputeFullV);
//     Vector4d pHom = svd.matrixV().block<4,1>(0,3);

// //    cout << "pHom" << endl << pHom << endl;
//     Vector3d p = pHom.block<3,1>(0,0) / pHom(3);
//     return p;
// }

void FeatureTracker::trackLandmarks(const Mat &image, int cameraNumber) {
    if (previousImages.empty()) return;
    if (landmarks.empty()) return;

    vector<Point2f> oldPoints;
    for (const auto & feature: landmarks) {
        oldPoints.emplace_back(feature.camCoordinates[cameraNumber]);
    }

    vector<Point2f> points;
    vector<uchar> status;
    vector<float> err;
    calcOpticalFlowPyrLK(previousImages[cameraNumber], image, oldPoints, points, status, err);

    for (long int i=points.size()-1; i >= 0; --i) {
        if (status[i] == 0) {
            landmarks.erase(landmarks.begin() + i);
        } else {
            landmarks[i].camCoordinates[cameraNumber] = points[i];
        }
    }
}

bool FeatureTracker::checkStereoQuality(const Point2f &leftKp, const Point2f &rightKp) {
    switch(mode) {
        case MODE::MULTIVIEW : {
            Vector3d homLeft, homRight;
            homLeft << leftKp.x, leftKp.y, 1;
            homRight << rightKp.x, rightKp.y, 1;

            double quality = (homLeft.transpose() * fundamentalMatrix * homRight);
            return (quality <= fundamentalThreshold);
        }
        case MODE::STEREO : {
            return (abs(leftKp.y - rightKp.y) <= stereoThreshold);
        }
        default : {
            return true;
        }
    }
}

vector<Landmark> FeatureTracker::matchImageFeatures(vector<vector<Point2f>> features, vector<Mat> images) const {
    vector<Landmark> foundLandmarks;
    for (int i=0; i<features[0].size(); ++i) {
        Landmark lm;
        Point2f proposedFeature = features[0][i];
        lm.camCoordinates.emplace_back(proposedFeature);

        if (mode == MODE::STEREO) {
            // TODO
        }

        foundLandmarks.emplace_back(lm);
    }

    return landmarks;
}

MODE::TrackerMode FeatureTracker::readMode(String modeName) {
//    if (modeName == "monocular") {
//        return MONO;
//    }
    if (modeName == "mono") {
        return MODE::MONO;
    }
    if (modeName == "stereo") {
        return MODE::STEREO;
    }
    if (modeName == "multiview") {
        return MODE::MULTIVIEW;
    }
    return MODE::MULTIVIEW;
}

void FeatureTracker::setCameraConfiguration(int cameraNumber, const CameraParameters &configuration) {
    assert(!(mode == MODE::MONO));
    assert(cameraNumber <= cameras.size());
    if (cameras.size() > cameraNumber) {
        cameras[cameraNumber] = configuration;
    } else {
        // allow cameras to be added via this method
        cameras.emplace_back(configuration);
    }
}


FeatureTracker::FeatureTracker(MODE::TrackerMode mode) {
    this->mode = mode;
}

vector<Point2f> FeatureTracker::detectNewFeatures(const Mat &image, int cameraNumber) const {
    // Obtain possible features
    Mat imageGrey;
    cv::cvtColor(image, imageGrey, cv::COLOR_BGR2GRAY);

    vector<Point2f> proposedFeatures;
    goodFeaturesToTrack(image, proposedFeatures, 2*numFeatures, 0.001, featureDist);

    // Remove duplicate features
    vector<Point2f> newFeatures;
    for (const auto & proposedFeature : proposedFeatures) {
        bool useFlag = true;
        for (const auto & feature : this->landmarks) {
            if (norm(proposedFeature - feature.camCoordinates[cameraNumber]) < featureDist) {
                useFlag = false;
                break;
            }
        }

        if (useFlag) {
            newFeatures.emplace_back(proposedFeature);
        }
    }

    return newFeatures;
}

void FeatureTracker::addNewLandmarks(vector<Landmark> newLandmarks) {
    for (auto & lm : newLandmarks) {
        lm.idNumber = ++currentNumber;
        landmarks.emplace_back(lm);
    }
}


Eigen::Matrix3d skew_matrix(const Eigen::Vector3d& t){
    Eigen::Matrix3d t_hat;
    t_hat << 0, -t(2), t(1),
            t(2), 0, -t(0),
            -t(1), t(0), 0;
    return t_hat;
}
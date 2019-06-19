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

using namespace GFT;

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
    if (mode == TrackerMode::STEREO) {
        newFeatures = this->detectNewStereoFeatures(images[0], images[1]);
    } else {
        for (int i=0; i<images.size(); ++i) {
            vector<Point2f> cameraFeatures = this->detectNewFeatures(images[i]);
            newFeatures.emplace_back(cameraFeatures);
        }
    }

    // Normalise new features
    vector<vector<Point2f>> newFeaturesNorm;
    for (int i=0; i<cameras.size(); ++i) {
        vector<Point2f> cameraFeaturesNorm;
        cv::undistortPoints(newFeatures[i], cameraFeaturesNorm, cameras[i].K, cameras[i].distortionParams);
        newFeaturesNorm.emplace_back(cameraFeaturesNorm);
    }

    // Compute positions and add to state
    vector<Landmark> newLandmarks = this->matchImageFeatures(newFeatures, newFeaturesNorm, images);
    this->addNewLandmarks(newLandmarks);
    this->computeLandmarkPositions();

}

void FeatureTracker::computeLandmarkPositions() {
    for (auto & lm : landmarks) {
        if (mode == TrackerMode::MONO) {
            lm.position << lm.camCoordinatesNorm[0].x, lm.camCoordinatesNorm[0].y, 1;
            lm.position.normalize();
        }

        if (mode == TrackerMode::STEREO) {
            lm.position = this->solveStereo(lm.camCoordinatesNorm[0], lm.camCoordinatesNorm[1]);
        }
    }
}

Vector3d FeatureTracker::solveStereo(const Point2f& leftKp, const Point2f& rightKp) const {
    assert(leftKp.x > rightKp.x);
    Vector3d position;
    position << leftKp.x, leftKp.y, 1;

    double scale = stereoBaseline / (leftKp.x - rightKp.x);
    position = scale*position;

    return position;
}

Vector3d FeatureTracker::solveMultiView(const vector<Point2f> imageCoordinatesNorm) const {
    int camNum = cameras.size();
    assert(imageCoordinatesNorm.size() == cameras.size());
    assert(camNum >= 2);

    MatrixXd solutionMat(3*camNum, 4);

    for (int i=0; i<camNum; ++i) {
        Vector3d imageCoord;
        imageCoord << imageCoordinatesNorm[i].x, imageCoordinatesNorm[i].y, 1;
        solutionMat.block<3,4>(3*i,0) = skew_matrix(imageCoord) * cameras[i].P;
    }

    JacobiSVD<MatrixXd> svd(solutionMat, ComputeThinU | ComputeFullV);
    Vector4d positionHomogeneous = svd.matrixV().block<4,1>(0,3);

    Vector3d position = positionHomogeneous.block<3,1>(0,0) / positionHomogeneous(3);
    return position;
}

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

    vector<Point2f> pointsNorm;
    cv::undistortPoints(points, pointsNorm, cameras[cameraNumber].K, cameras[cameraNumber].distortionParams);

    for (long int i=points.size()-1; i >= 0; --i) {
        bool eraseCondition = (status[i] == 0);
        if (!imageMasks[cameraNumber].empty()) {
            eraseCondition |= imageMasks[cameraNumber].at<uchar>(points[i]);
        }

        if (eraseCondition) {
            landmarks.erase(landmarks.begin() + i);
        } else {
            landmarks[i].camCoordinates[cameraNumber] = points[i];
            landmarks[i].camCoordinatesNorm[cameraNumber] = pointsNorm[i];
            ++landmarks[i].lifetime;
        }
    }
}

vector<Landmark> FeatureTracker::matchImageFeatures(vector<vector<Point2f>> features, vector<vector<Point2f>> featuresNorm, vector<Mat> images) const {
    vector<Landmark> foundLandmarks;
    for (int i=0; i<features[0].size(); ++i) {
        Landmark lm;
        Point2f proposedFeature = features[0][i];
        Point2f proposedFeatureNorm = features[0][i];
        lm.camCoordinates.emplace_back(proposedFeature);
        lm.camCoordinatesNorm.emplace_back(proposedFeatureNorm);

        if (mode == TrackerMode::STEREO) {
            // TODO
        }

        foundLandmarks.emplace_back(lm);
    }

    return foundLandmarks;
}

void FeatureTracker::setCameraConfiguration(const CameraParameters &configuration, int cameraNumber) {
    assert(cameraNumber == 0 || !(mode == TrackerMode::MONO));
    assert(cameraNumber <= cameras.size());
    if (cameras.size() > cameraNumber) {
        cameras[cameraNumber] = configuration;
    } else {
        // allow cameras to be added via this method
        cameras.emplace_back(configuration);
        Mat emptyMask;
        imageMasks.emplace_back(emptyMask);
    }   
}


FeatureTracker::FeatureTracker(TrackerMode mode) {
    this->mode = mode;

}

vector<Point2f> FeatureTracker::detectNewFeatures(const Mat &image, int cameraNumber) const {
    Mat imageGrey;
    cv::cvtColor(image, imageGrey, cv::COLOR_BGR2GRAY);

    vector<Point2f> proposedFeatures;
    goodFeaturesToTrack(imageGrey, proposedFeatures, 2*maxFeatures, 0.001, featureDist, imageMasks[cameraNumber]);
    vector<Point2f> newFeatures = this->removeDuplicateFeatures(proposedFeatures, cameraNumber);

    return newFeatures;
}

vector<vector<Point2f>> FeatureTracker::detectNewStereoFeatures(const cv::Mat &imageLeft, const cv::Mat &imageRight) const {
    vector<vector<Point2f>> newFeatures(2);

    // Obtain left features
    Mat imageGreyLeft;
    cv::cvtColor(imageLeft, imageGreyLeft, cv::COLOR_BGR2GRAY);
    goodFeaturesToTrack(imageGreyLeft, newFeatures[0], 2*maxFeatures, 0.001, featureDist, imageMasks[0]);
    newFeatures[0] = this->removeDuplicateFeatures(newFeatures[0]);

    // Track features to the right image
    vector<uchar> status;
    vector<float> err;
    calcOpticalFlowPyrLK(imageLeft, imageRight, newFeatures[0], newFeatures[1], status, err);

    assert(newFeatures[0].size() == newFeatures[1].size());

    for (long int i=newFeatures[0].size()-1; i >= 0; --i) {
        bool eraseCondition = (status[i] == 0);
        if (!imageMasks[0].empty()) eraseCondition |= imageMasks[0].at<uchar>(newFeatures[0][i]);
        if (!imageMasks[1].empty()) eraseCondition |= imageMasks[1].at<uchar>(newFeatures[1][i]);

        if (eraseCondition) {
            newFeatures[0].erase(newFeatures[0].begin() +i);
            newFeatures[1].erase(newFeatures[1].begin() +i);
        }
    }

    return newFeatures;

}

vector<Point2f> FeatureTracker::removeDuplicateFeatures(const vector<Point2f> &proposedFeatures, int cameraNumber) const {
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
        if (landmarks.size() >= maxFeatures) break;

        lm.idNumber = ++currentNumber;
        landmarks.emplace_back(lm);
    }
}

void FeatureTracker::setMask(const Mat & mask, int cameraNumber) {
    imageMasks[cameraNumber] = mask;
}


void FeatureTracker::setMasks(const vector<Mat> & masks) {
    assert(masks.size() == imageMasks.size());
    imageMasks = masks;
}



Eigen::Matrix3d GFT::skew_matrix(const Eigen::Vector3d& t){
    Eigen::Matrix3d t_hat;
    t_hat << 0, -t(2), t(1),
            t(2), 0, -t(0),
            -t(1), t(0), 0;
    return t_hat;
}
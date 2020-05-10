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

#include <FeatureTracker.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/stereo/stereo.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "eigen3/Eigen/SVD"
#include "iostream"
#include "string"

using namespace GIFT;

void FeatureTracker::processImage(const Mat &image) {
    this->trackLandmarks(image);
    image.copyTo(this->previousImage);

    if (this->landmarks.size() > this->featureSearchThreshold*this->maxFeatures) return;

    vector<Point2f> newFeatures = this->detectNewFeatures(image);
    vector<Landmark> newLandmarks = this->createNewLandmarks(image, newFeatures);

    this->addNewLandmarks(newLandmarks);
}

vector<Landmark> FeatureTracker::createNewLandmarks(const Mat &image, const vector<Point2f>& newFeatures) {
    vector<Landmark> newLandmarks;
    if (newFeatures.empty()) return newLandmarks;

    vector<Point2f> newFeaturesNorm;
    cv::undistortPoints(newFeatures, newFeaturesNorm, camera.K, camera.distortionParams);

    for (int i = 0; i < newFeatures.size(); ++i) {
        
        Point2f proposedFeature = newFeatures[i];
        Point2f proposedFeatureNorm = newFeaturesNorm[i];

        colorVec pointColor = {image.at<Vec3b>(newFeatures[i]).val[0],
                               image.at<Vec3b>(newFeatures[i]).val[1],
                               image.at<Vec3b>(newFeatures[i]).val[2]};

        Landmark lm(proposedFeature, proposedFeatureNorm, -1, pointColor);

        newLandmarks.emplace_back(lm);
    }

    return newLandmarks;
}

void FeatureTracker::trackLandmarks(const Mat &image) {
    if (landmarks.empty()) return;

    vector<Point2f> oldPoints;
    for (const auto & feature: landmarks) {
        oldPoints.emplace_back(feature.camCoordinates);
    }

    vector<Point2f> points;
    vector<uchar> status;
    vector<float> err;
    calcOpticalFlowPyrLK(previousImage, image, oldPoints, points, status, err);

    vector<Point2f> pointsNorm;
    cv::undistortPoints(points, pointsNorm, camera.K, camera.distortionParams);

    for (long int i=points.size()-1; i >= 0; --i) {
        if (status[i] == 0) {
            landmarks.erase(landmarks.begin() + i);
            continue;
        }

        if (!imageMask.empty()) {
            if (imageMask.at<uchar>(points[i])==0) {
                landmarks.erase(landmarks.begin() + i);
                continue;
            } 
        }

        colorVec pointColor = {image.at<Vec3b>(points[i]).val[0],
                                image.at<Vec3b>(points[i]).val[1],
                                image.at<Vec3b>(points[i]).val[2]};
        landmarks[i].update(points[i], pointsNorm[i], pointColor);
        
    }
}

void FeatureTracker::setCameraConfiguration(const CameraParameters &configuration) {
    camera = configuration;
}

vector<Point2f> FeatureTracker::detectNewFeatures(const Mat &image) const {
    Mat imageGrey;
    cv::cvtColor(image, imageGrey, cv::COLOR_BGR2GRAY);

    vector<Point2f> proposedFeatures;
    goodFeaturesToTrack(imageGrey, proposedFeatures, maxFeatures, minHarrisQuality, featureDist, imageMask);
    vector<Point2f> newFeatures = this->removeDuplicateFeatures(proposedFeatures);

    return newFeatures;
}

vector<Point2f> FeatureTracker::removeDuplicateFeatures(const vector<Point2f> &proposedFeatures) const {
    vector<Point2f> newFeatures;
    for (const auto & proposedFeature : proposedFeatures) {
        bool useFlag = true;
        for (const auto & feature : this->landmarks) {
            if (norm(proposedFeature - feature.camCoordinates) < featureDist) {
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
    imageMask = mask;
}



Eigen::Matrix3T GIFT::skew_matrix(const Eigen::Vector3T& t){
    Eigen::Matrix3T t_hat;
    t_hat << 0, -t(2), t(1),
            t(2), 0, -t(0),
            -t(1), t(0), 0;
    return t_hat;
}

Mat FeatureTracker::drawFeatureImage(const Scalar& color, const int pointSize, const int thickness) const {
        cv::Mat featureImage;
        this->previousImage.copyTo(featureImage);
        for (const auto &lm : this->landmarks) {
            cv::circle(featureImage, lm.camCoordinates, pointSize, color, thickness);
        }
        return featureImage;
}

Mat FeatureTracker::drawFlowImage(const Scalar& featureColor, const Scalar& flowColor, const int pointSize, const int thickness) const {
    Mat flowImage = drawFeatureImage(featureColor, pointSize, thickness);
    for (const auto &lm : this->landmarks) {
            Point2f p1 = lm.camCoordinates;
            Point2f p0 =  p1 - Point2f(lm.opticalFlowRaw.x(), lm.opticalFlowRaw.y());
            line(flowImage, p0, p1, flowColor, thickness);
        }
    return flowImage;
}

Mat FeatureTracker::drawFlow(const Scalar& featureColor, const Scalar& flowColor, const int pointSize, const int thickness) const {
    Mat flow(this->previousImage.size(), CV_8UC3);
    flow.setTo(0);
    
    for (const auto &lm : this->landmarks) {
            Point2f p1 = lm.camCoordinates;
            Point2f p0 =  p1 - Point2f(lm.opticalFlowRaw.x(), lm.opticalFlowRaw.y());
            circle(flow, p1, pointSize, featureColor, thickness);
            line(flow, p0, p1, flowColor, thickness);
        }
    return flow;
}


/*
void FeatureTracker::computeLandmarkPositions() {
    if (mode == TrackerMode::MONO) return;
    for (auto & lm : landmarks) {
        if (mode == TrackerMode::STEREO) {
            lm.position = this->solveStereo(lm.camCoordinatesNorm[0], lm.camCoordinatesNorm[1]);
        }
    }
}

Vector3T FeatureTracker::solveStereo(const Point2f& leftKp, const Point2f& rightKp) const {
    assert(leftKp.x > rightKp.x);
    Vector3T position;
    position << leftKp.x, leftKp.y, 1;

    ftype scale = stereoBaseline / (leftKp.x - rightKp.x);
    position = scale*position;

    return position;
}

Vector3T FeatureTracker::solveMultiView(const vector<Point2f> imageCoordinatesNorm) const {
    int camNum = cameras.size();
    assert(imageCoordinatesNorm.size() == cameras.size());
    assert(camNum >= 2);

    MatrixXT solutionMat(3*camNum, 4);

    for (int i=0; i<camNum; ++i) {
        Vector3T imageCoord;
        imageCoord << imageCoordinatesNorm[i].x, imageCoordinatesNorm[i].y, 1;
        solutionMat.block<3,4>(3*i,0) = skew_matrix(imageCoord) * cameras[i].P;
    }

    JacobiSVD<MatrixXT> svd(solutionMat, ComputeThinU | ComputeFullV);
    Vector4T positionHomogeneous = svd.matrixV().block<4,1>(0,3);

    Vector3T position = positionHomogeneous.block<3,1>(0,0) / positionHomogeneous(3);
    return position;
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
*/

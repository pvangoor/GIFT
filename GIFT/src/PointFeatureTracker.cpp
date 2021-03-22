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

#include "eigen3/Eigen/SVD"
#include "iostream"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stereo/stereo.hpp"
#include "opencv2/video/tracking.hpp"
#include "string"
#include <PointFeatureTracker.h>

using namespace GIFT;
using namespace Eigen;
using namespace std;
using namespace cv;

void PointFeatureTracker::processImage(const Mat& image) {
    this->trackFeatures(image);
    image.copyTo(this->previousImage);

    if (this->features.size() > this->featureSearchThreshold * this->maxFeatures)
        return;

    detectFeatures(image);
}

void PointFeatureTracker::detectFeatures(const Mat& image) {
    vector<Point2f> newPoints = this->identifyFeatureCandidates(image);
    vector<Feature> newFeatures = this->createNewFeatures(image, newPoints);
    this->addNewFeatures(newFeatures);
}

vector<Feature> PointFeatureTracker::createNewFeatures(const Mat& image, const vector<Point2f>& newPoints) {
    vector<Feature> newFeatures;
    if (newPoints.empty())
        return newFeatures;

    for (int i = 0; i < newPoints.size(); ++i) {
        colorVec pointColor = {image.at<Vec3b>(newPoints[i]).val[0], image.at<Vec3b>(newPoints[i]).val[1],
            image.at<Vec3b>(newPoints[i]).val[2]};

        Feature lm(newPoints[i], cameraPtr, -1, pointColor);

        newFeatures.emplace_back(lm);
    }

    return newFeatures;
}

void PointFeatureTracker::trackFeatures(const Mat& image) {
    if (features.empty())
        return;

    vector<Point2f> oldPoints;
    for (const auto& feature : features) {
        oldPoints.emplace_back(feature.camCoordinates);
    }

    vector<Point2f> points;
    vector<uchar> status;
    vector<float> err;
    calcOpticalFlowPyrLK(previousImage, image, oldPoints, points, status, err, Size(winSize, winSize), maxLevel,
        TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 30, (0.01000000000000000021)));

    for (long int i = points.size() - 1; i >= 0; --i) {
        if (status[i] == 0 || err[i] >= maxError) {
            features.erase(features.begin() + i);
            continue;
        }

        if (!mask.empty()) {
            if (mask.at<uchar>(points[i]) == 0) {
                features.erase(features.begin() + i);
                continue;
            }
        }

        colorVec pointColor = {
            image.at<Vec3b>(points[i]).val[0], image.at<Vec3b>(points[i]).val[1], image.at<Vec3b>(points[i]).val[2]};
        features[i].update(points[i], pointColor);
    }
}

vector<Point2f> PointFeatureTracker::identifyFeatureCandidates(const Mat& image) const {
    Mat imageGrey;
    if (image.channels() > 1)
        [[unlikely]] { cv::cvtColor(image, imageGrey, cv::COLOR_BGR2GRAY); }
    else {
        imageGrey = image;
    }

    vector<Point2f> proposedFeatures;
    goodFeaturesToTrack(imageGrey, proposedFeatures, maxFeatures, minHarrisQuality, featureDist, mask);
    vector<Point2f> newFeatures = this->removeDuplicateFeatures(proposedFeatures);

    return newFeatures;
}

vector<Point2f> PointFeatureTracker::removeDuplicateFeatures(const vector<Point2f>& proposedFeatures) const {
    vector<Point2f> newFeatures;
    for (const auto& proposedFeature : proposedFeatures) {
        bool useFlag = true;
        for (const auto& feature : this->features) {
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

void PointFeatureTracker::addNewFeatures(vector<Feature> newFeatures) {
    for (auto& lm : newFeatures) {
        if (features.size() >= maxFeatures)
            break;

        lm.idNumber = ++currentNumber;
        features.emplace_back(lm);
    }
}

Eigen::Matrix3T GIFT::skew_matrix(const Eigen::Vector3T& t) {
    Eigen::Matrix3T t_hat;
    t_hat << 0, -t(2), t(1), t(2), 0, -t(0), -t(1), t(0), 0;
    return t_hat;
}

void PointFeatureTracker::useFeaturePredictions(const std::vector<Feature>& predictedFeatures) {
    for (const Feature& pf : predictedFeatures) {
        // Find a match
        for (Feature& f : this->features) {
            if (f.idNumber == pf.idNumber) {
                f.camCoordinates = pf.camCoordinates;
                break;
            }
        }
    }
}
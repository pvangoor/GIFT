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

#include "KeyPointFeatureTracker.h"


void GIFT::KeyPointFeatureTracker::detectFeatures(const Mat &image) {
    // Detect features and compute descriptors
    vector<KeyPoint> newKeypoints;
    Mat newDescriptors;
    ORBDetector->detectAndCompute(image, mask, newKeypoints, newDescriptors);

    // Construct features
    vector<InternalKPFeature> newFeatures(newKeypoints.size());
    transform(newKeypoints.begin(), newKeypoints.end(), newFeatures.begin(), [](const KeyPoint& kp) {
        InternalKPFeature result;
        result.kp = kp;
        result.lifetime = 0;
        return result;
    }
    );

    // Set descriptors
    for (int i=0; i<newFeatures.size(); ++i){
        InternalKPFeature& feature = newFeatures[i];
        feature.descriptor = newDescriptors.row(i);
    }

    // Remove points and set up id numbers
    removePointsTooClose(newFeatures);
    const int pointsToAdd = settings.maximumFeatures - features.size();
    newFeatures.resize(max(0,pointsToAdd));
    for_each(newFeatures.begin(), newFeatures.end(), [this](InternalKPFeature& f) {f.id = ++this->currentNumber; });

    // Add the features to the current feature list
    features.insert(features.end(), newFeatures.begin(), newFeatures.end());
}

void GIFT::KeyPointFeatureTracker::trackFeatures(const Mat &image) {
    // Detect features and compute descriptors
    vector<KeyPoint> newKeypoints; Mat newDescriptors;
    ORBDetector->detectAndCompute(image, mask, newKeypoints, newDescriptors);

    // Construct current features descriptor matrix
    vector<Mat> descriptorRows(features.size());
    transform(features.begin(), features.end(), descriptorRows.begin(), [](const InternalKPFeature& f) {return f.descriptor;});
    Mat descriptors; vconcat(descriptorRows, descriptors);

    // Match the current features to the detected features
    vector<DMatch> matches;
    matcher->match(descriptors, newDescriptors, matches);
    for (const DMatch& match : matches) {
        InternalKPFeature& feature = features[match.queryIdx];
        ++feature.lifetime;
        feature.kp = newKeypoints[match.trainIdx];
    }

}

vector<GIFT::Landmark> GIFT::KeyPointFeatureTracker::outputLandmarks() const {
    vector<Landmark> landmarks(features.size());
    transform(features.begin(), features.end(), landmarks.begin(), [this](const InternalKPFeature& f){
        return this->featureToLandmark(f);
    });
    return landmarks;
}

GIFT::Landmark GIFT::KeyPointFeatureTracker::featureToLandmark(const InternalKPFeature& feature) const {
    GIFT::Landmark lm;
    lm.camCoordinates = feature.camCoordinates();
    // undistortPoints(lm.camCoordinates, lm.camCoordinatesNorm, camera.K, camera.distortionParams);
    lm.idNumber = feature.id;
    lm.lifetime = feature.lifetime;
    // lm.pointColor.fill();
    return lm;
    // TODO: Some parts of the landmark are missing. Is this a problem?
}

void GIFT::KeyPointFeatureTracker::removePointsTooClose(vector<InternalKPFeature>& newFeatures) const {
    const double minDistSq = settings.minimumFeatureDistance*settings.minimumFeatureDistance;
    for (int i=newFeatures.size()-1; i>=0; --i) {
        const Point2f& newPoint = newFeatures[i].kp.pt;
        for (const InternalKPFeature& f : features) {
            const Point2f& point = f.kp.pt;
            const double distSq = (point.x - newPoint.x)*(point.x - newPoint.x) + (point.y - newPoint.y)*(point.y - newPoint.y);
            if (distSq < minDistSq) {
                newFeatures.erase(newFeatures.begin()+i);
                break;
            }
        }
    }
}
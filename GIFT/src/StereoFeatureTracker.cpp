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

#include "GIFT/StereoFeatureTracker.h"
#include <set>

using namespace GIFT;
using namespace Eigen;
using namespace std;
using namespace cv;

void StereoFeatureTracker::processImages(const Mat& imageLeft, const Mat& imageRight) {
    trackerLeft.processImage(imageLeft);
    trackerRight.processImage(imageRight);

    const vector<Feature> featuresLeft = trackerLeft.outputFeatures();
    const vector<Feature> featuresRight = trackerRight.outputFeatures();

    removeLostStereoFeatures(featuresLeft, featuresRight);
    vector<StereoLandmark> newStereoFeatures =
        createNewStereoFeatures(featuresLeft, imageLeft, featuresRight, imageRight);
    addNewStereoFeatures(newStereoFeatures);
}

void StereoFeatureTracker::removeLostStereoFeatures(
    const vector<Feature>& featuresLeft, const vector<Feature>& featuresRight) {
    set<int> idsLeft, idsRight;
    for (const Feature& lm : featuresLeft)
        idsLeft.emplace(lm.idNumber);
    for (const Feature& lm : featuresRight)
        idsRight.emplace(lm.idNumber);

    auto checkValidLandmark = [idsLeft, idsRight](const int idLeft, const int idRight) {
        return (idsLeft.count(idLeft) * idsRight.count(idRight));
    };

    vector<StereoLandmark>::iterator iter = stereoFeatures.begin();
    while (iter != stereoFeatures.end()) {
        if (checkValidLandmark(iter->featureLeft->idNumber, iter->featureRight->idNumber)) {
            ++iter;
        } else {
            iter = stereoFeatures.erase(iter);
        }
    }
}

vector<StereoLandmark> StereoFeatureTracker::createNewStereoFeatures(const vector<Feature>& featuresLeft,
    const Mat& imageLeft, const vector<Feature>& featuresRight, const Mat& imageRight) const {
    vector<StereoLandmark> newFeatures;
    return newFeatures;
}

void StereoFeatureTracker::addNewStereoFeatures(const vector<StereoLandmark>& newStereoFeatures) {
    set<int> idsLeft, idsRight;
    for (const StereoLandmark& lm : this->stereoFeatures) {
        idsLeft.emplace(lm.featureLeft->idNumber);
        idsRight.emplace(lm.featureRight->idNumber);
    }

    auto checkValidLandmark = [idsLeft, idsRight](const int idLeft, const int idRight) {
        return (idsLeft.count(idLeft) * idsRight.count(idRight));
    };

    for (const StereoLandmark& stereoLM : newStereoFeatures) {
        if (checkValidLandmark(stereoLM.featureLeft->idNumber, stereoLM.featureRight->idNumber)) {
            this->stereoFeatures.emplace_back(stereoLM);
        }
    }
}
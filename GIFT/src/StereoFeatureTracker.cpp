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

#include "StereoFeatureTracker.h"
#include <set>

using namespace GIFT;
using namespace Eigen;
using namespace std;
using namespace cv;

void StereoFeatureTracker::processImages(const Mat& imageLeft, const Mat& imageRight) {
    trackerLeft.processImage(imageLeft);
    trackerRight.processImage(imageRight);

    const vector<Feature> landmarksLeft = trackerLeft.outputLandmarks();
    const vector<Feature> landmarksRight = trackerRight.outputLandmarks();

    removeLostStereoLandmarks(landmarksLeft, landmarksRight);
    vector<StereoLandmark> newStereoLandmarks =
        createNewStereoLandmarks(landmarksLeft, imageLeft, landmarksRight, imageRight);
    addNewStereoLandmarks(newStereoLandmarks);
}

void StereoFeatureTracker::removeLostStereoLandmarks(
    const vector<Feature>& landmarksLeft, const vector<Feature>& landmarksRight) {
    set<int> idsLeft, idsRight;
    for (const Feature& lm : landmarksLeft)
        idsLeft.emplace(lm.idNumber);
    for (const Feature& lm : landmarksRight)
        idsRight.emplace(lm.idNumber);

    auto checkValidLandmark = [idsLeft, idsRight](const int idLeft, const int idRight) {
        return (idsLeft.count(idLeft) * idsRight.count(idRight));
    };

    vector<StereoLandmark>::iterator iter = stereoLandmarks.begin();
    while (iter != stereoLandmarks.end()) {
        if (checkValidLandmark(iter->landmarkLeft->idNumber, iter->landmarkRight->idNumber)) {
            ++iter;
        } else {
            iter = stereoLandmarks.erase(iter);
        }
    }
}

vector<StereoLandmark> StereoFeatureTracker::createNewStereoLandmarks(const vector<Feature>& landmarksLeft,
    const Mat& imageLeft, const vector<Feature>& landmarksRight, const Mat& imageRight) const {
    vector<StereoLandmark> newLandmarks;
    return newLandmarks;
}

void StereoFeatureTracker::addNewStereoLandmarks(const vector<StereoLandmark>& newStereoLandmarks) {
    set<int> idsLeft, idsRight;
    for (const StereoLandmark& lm : this->stereoLandmarks) {
        idsLeft.emplace(lm.landmarkLeft->idNumber);
        idsRight.emplace(lm.landmarkRight->idNumber);
    }

    auto checkValidLandmark = [idsLeft, idsRight](const int idLeft, const int idRight) {
        return (idsLeft.count(idLeft) * idsRight.count(idRight));
    };

    for (const StereoLandmark& stereoLM : newStereoLandmarks) {
        if (checkValidLandmark(stereoLM.landmarkLeft->idNumber, stereoLM.landmarkRight->idNumber)) {
            this->stereoLandmarks.emplace_back(stereoLM);
        }
    }
}
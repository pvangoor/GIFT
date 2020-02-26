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
#include "opencv2/video/tracking.hpp"

using namespace GIFT;

void StereoFeatureTracker::processImages(const Mat &imageLeft, const Mat &imageRight) {
    trackerLeft.processImage(imageLeft);
    trackerRight.processImage(imageRight);

    vector<Landmark> landmarksLeft = trackerLeft.outputLandmarks();
    vector<Landmark> landmarksRight = trackerRight.outputLandmarks();

    removeLostStereoLandmarks(landmarksLeft, landmarksRight);
    vector<StereoLandmark> newStereoLandmarks = createNewStereoLandmarks(landmarksLeft, imageLeft, landmarksRight, imageRight);
    // addNewStereoLandmarks(newStereoLandmarks);
    stereoLandmarks = newStereoLandmarks;
}

void StereoFeatureTracker::removeLostStereoLandmarks(const vector<Landmark>& landmarksLeft, const vector<Landmark>& landmarksRight) {
    set<int> idsLeft, idsRight;
    for (const Landmark& lm : landmarksLeft) idsLeft.emplace(lm.idNumber);
    for (const Landmark& lm : landmarksRight) idsRight.emplace(lm.idNumber);

    auto checkValidLandmark = [idsLeft, idsRight] (const int idLeft, const int idRight) {return (idsLeft.count(idLeft) * idsRight.count(idRight)); };

    vector<StereoLandmark>::iterator iter = stereoLandmarks.begin();
    while (iter != stereoLandmarks.end()) {
        if (checkValidLandmark(iter->landmarkLeft->idNumber, iter->landmarkRight->idNumber)) {
            ++iter;
        } else {
            iter = stereoLandmarks.erase(iter);
        }
    }
}

vector<StereoLandmark> StereoFeatureTracker::createNewStereoLandmarks(vector<Landmark>& landmarksLeft, const Mat& imageLeft,
                                                                      vector<Landmark>& landmarksRight, const Mat& imageRight) const {
    vector<StereoLandmark> newLandmarks = findStereoLandmarks(landmarksLeft, imageLeft, imageRight);
    return newLandmarks;
}
 
void StereoFeatureTracker::addNewStereoLandmarks(const vector<StereoLandmark>& newStereoLandmarks) {
    set<int> idsLeft, idsRight;
    for (const StereoLandmark& lm : this->stereoLandmarks){
        idsLeft.emplace(lm.landmarkLeft->idNumber);
        idsRight.emplace(lm.landmarkRight->idNumber);
    }

    auto checkValidLandmark = [idsLeft, idsRight] (const int idLeft, const int idRight) {return (idsLeft.count(idLeft) * idsRight.count(idRight)); };

    for (const StereoLandmark& stereoLM : newStereoLandmarks) {
        if (checkValidLandmark(stereoLM.landmarkLeft->idNumber, stereoLM.landmarkRight->idNumber)) {
            this->stereoLandmarks.emplace_back(stereoLM);
        }
    }
}

vector<StereoLandmark> StereoFeatureTracker::findStereoLandmarks(
vector<Landmark>& landmarksLeft,
const Mat& imageLeft, const Mat& imageRight,
Size winSize, const int maxLevel) const {
    // Construct pyramids to calculate the optical flow

    Mat imageLeftGrey, imageRightGrey;
    cv::cvtColor(imageLeft, imageLeftGrey, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imageRight, imageRightGrey, cv::COLOR_BGR2GRAY);
    vector<Mat> pyramidLeft(maxLevel+1), pyramidRight(maxLevel+1);
    for (int i=0; i <= maxLevel; ++i ) {
        if (i > 0) {
            pyrDown(pyramidLeft[i-1], pyramidLeft[i]);
            pyrDown(pyramidRight[i-1], pyramidRight[i]);
        } else {
            pyramidLeft[0] = imageLeftGrey;
            pyramidRight[0] = imageRightGrey;
        }
    }
    for (int i=0; i<=maxLevel; ++i) {
        copyMakeBorder(pyramidLeft[i], pyramidLeft[i], (winSize.height/2), (winSize.height/2), (winSize.width/2), (winSize.width/2), BORDER_REPLICATE);
        copyMakeBorder(pyramidRight[i], pyramidRight[i], (winSize.height/2), (winSize.height/2), (winSize.width/2), (winSize.width/2), BORDER_REPLICATE);
    }

    // Precompute
    int n = landmarksLeft.size();
    int vecSize = winSize.area();

    // Initialise x offsets to zero
    vector<double> offsets(n);
    vector<bool> convergence(n);
    for (int i=0;i<n;++i) offsets[i] = 0.0;


    // Iterate over pyramid levels to find stereo matches, starting at the top.
    for (int lv=maxLevel; lv >= 0; --lv) {
        // Retrieve the pyramid level, and compute the gradient
        Mat pyrImgL = pyramidLeft[lv];
        Mat pyrImgR = pyramidRight[lv];
        Rect2f pyrBound = Rect2f(0,0, pyrImgR.cols-winSize.width-1, pyrImgR.rows-winSize.height-1);

        Mat dx_pyrImgL;
        Sobel( pyrImgL, dx_pyrImgL, CV_16SC1, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);

        // Find the stereo match for each point.
        for (int i=0; i<n; ++i) {
            Landmark landmark = landmarksLeft[i];
            Point2f pointL = landmark.camCoordinates;
            // Adjust coordinates of the point to match the pyramid level.
            // The padding is taken care of in the rectangle definition.
            pointL = pointL * pow(2,-lv);
            if (! pyrBound.contains(pointL)) continue;

            // Retrieve the image and gradient data
            Rect2f rectL = Rect2f(pointL.x, pointL.y, winSize.width, winSize.height);
            Mat patchL = pyrImgL(rectL);
            Mat dx_patchL = dx_pyrImgL(rectL);

            // Vectorise the image and gradient data
            Mat vec_patchL(vecSize, 1, CV_32FC1), vec_dx_patchL(vecSize, 1, CV_32FC1);
            for (int y=0; y<winSize.height; ++y) {
                for (int x=0;x<winSize.width; ++x) {
                    vec_patchL.at<float>(y*winSize.width+x, 0) = (float)patchL.at<uchar>(y,x);
                    vec_dx_patchL.at<float>(y*winSize.width+x, 0) = (float)dx_patchL.at<uchar>(y,x);
                }
            }
            Mat jacobianT = (vec_dx_patchL.t() * vec_dx_patchL).inv() * vec_dx_patchL.t();
            
            // Converge the new point
            bool converged = false;
            int stepCount = 0;
            double& offset = offsets[i];
            while (!converged) {
                Point2f pointR = pointL + Point2f(offset, 0.0);

                // Ensure the point has not left the image
                if (! pyrBound.contains(pointR)) break;
                
                // Get the patch and vectorise
                Rect2f rectR = Rect2f(pointR.x, pointR.y, winSize.width, winSize.height);
                Mat patchR = pyrImgR(rectR);
                Mat vec_patchR(vecSize, 1, CV_32FC1);
                for (int y=0; y<winSize.height; ++y) {
                    for (int x=0;x<winSize.width; ++x) {
                        vec_patchR.at<float>(y*winSize.width+x, 0) = (float)patchR.at<uchar>(y,x);
                    }
                }

                // Compute the error and the step
                Mat vec_error = vec_patchR - vec_patchL;
                Mat step = jacobianT * vec_error;

                // print(vec_error);
                // print(step);
                float stepf = -step.at<float>(0,0);
                offset = offset + stepf;
                if (abs(stepf) < 1e-3 || ++stepCount >= 100) {
                    converged = true;
                }

            }

            // If converged, increase the offset to the next level
            if (converged && lv > 0) offset = offset*2;
            else if (offset < 0) offset = 0;
            convergence[i] = converged;
        }

    }

    // With the offsets computed, create the new right landmarks
    
    vector<StereoLandmark> newLandmarks;
    for (int i=0;i<n;++i) {
        if (!convergence[i]) continue;
        Landmark newLandmarkRight;
        newLandmarkRight.camCoordinates = landmarksLeft[i].camCoordinates + Point2f(offsets[i], 0.0);
        StereoLandmark newStereoLandmark(landmarksLeft[i], newLandmarkRight, landmarksLeft[i].idNumber);
        newLandmarks.emplace_back(newStereoLandmark);
    }

    return newLandmarks;

}
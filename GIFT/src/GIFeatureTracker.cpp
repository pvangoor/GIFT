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
#include "GIFeatureTracker.h"


using namespace GIFT;


// Initialisation and configuration
GIFeatureTracker::GIFeatureTracker(const CameraParameters& cameraParams, const Mat& mask) {
    this->setCameraParameters(cameraParams);
    this->setMask(mask);
}
GIFeatureTracker::GIFeatureTracker(const CameraParameters& cameraParams) {
    this->setCameraParameters(cameraParams);
}

void GIFeatureTracker::setCameraParameters(const CameraParameters & cameraParameters) {
    this->camera = cameraParameters;
}

void GIFeatureTracker::setMask(const Mat & mask) {
    this->mask = mask;
}
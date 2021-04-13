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

#include "Feature.h"
#include "Camera.h"

using namespace GIFT;
using namespace cv;
using namespace Eigen;

Feature::Feature(
    const Point2f& newCamCoords, const std::shared_ptr<const GICamera>& cameraPtr, int idNumber, const colorVec& col) {
    this->camCoordinates = newCamCoords;
    this->cameraPtr = cameraPtr;

    this->opticalFlowRaw.setZero();
    this->opticalFlowNorm.setZero();
    this->opticalFlowSphere().setZero();

    this->pointColor = col;
    this->idNumber = idNumber;

    lifetime = 1;
}

void Feature::update(const cv::Point2f& newCamCoords, const colorVec& col) {
    this->opticalFlowRaw << newCamCoords.x - this->camCoordinates.x, newCamCoords.y - this->camCoordinates.y;
    cv::Point2f newCamCoordsNorm = cameraPtr->undistortPointCV(newCamCoords);
    this->opticalFlowNorm << newCamCoordsNorm.x - this->camCoordinatesNorm().x,
        newCamCoordsNorm.y - this->camCoordinatesNorm().y;

    this->camCoordinates = newCamCoords;

    Vector3T bearing = Vector3T(newCamCoordsNorm.x, newCamCoordsNorm.y, 1).normalized();

    this->opticalFlowSphere() = bearing.z() * (Matrix3T::Identity() - bearing * bearing.transpose()) *
                                Vector3T(opticalFlowNorm.x(), opticalFlowNorm.y(), 0);

    this->pointColor = col;
    ++lifetime;
}

Eigen::Vector3T Feature::sphereCoordinates() const {
    Eigen::Vector3T result;
    result << camCoordinatesNorm().x, camCoordinatesNorm().y, 1.0;
    return result.normalized();
}

cv::Point2f Feature::camCoordinatesNorm() const { return cameraPtr->undistortPointCV(camCoordinates); }

Eigen::Vector3T Feature::opticalFlowSphere() const {
    const Vector3T bearing = sphereCoordinates();
    const Vector3T sphereFlow = bearing.z() * (Matrix3T::Identity() - bearing * bearing.transpose()) *
                                Vector3T(opticalFlowNorm.x(), opticalFlowNorm.y(), 0);
    return sphereFlow;
}
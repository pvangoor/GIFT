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

#include "Landmark.h"

using namespace GIFT;
using namespace cv;
using namespace Eigen;


Landmark::Landmark(const Point2f& newCamCoords, const Point2f& newCamCoordsNorm, int idNumber, const colorVec& col) {
    this->camCoordinates = newCamCoords;
    this->camCoordinatesNorm = newCamCoordsNorm;
    this->sphereCoordinates << newCamCoordsNorm.x, newCamCoordsNorm.y, 1;
    this->sphereCoordinates.normalize();

    this->opticalFlowRaw.setZero();
    this->opticalFlowNorm.setZero();
    this->opticalFlowSphere.setZero();

    this->keypoint.pt = this->camCoordinates;

    this->pointColor = col;
    this->idNumber = idNumber;

    lifetime = 1;
}

void Landmark::update(const cv::Point2f& newCamCoords, const cv::Point2f& newCamCoordsNorm, const colorVec& col) {
    this->opticalFlowRaw << newCamCoords.x - this->camCoordinates.x, newCamCoords.y - this->camCoordinates.y;
    this->opticalFlowNorm << newCamCoordsNorm.x - this->camCoordinatesNorm.x, newCamCoordsNorm.y - this->camCoordinatesNorm.y;

    this->camCoordinates = newCamCoords;
    this->camCoordinatesNorm = newCamCoordsNorm;

    Vector3T bearing = Vector3T(newCamCoordsNorm.x, newCamCoordsNorm.y, 1).normalized();

    this->sphereCoordinates = bearing;
    this->opticalFlowSphere = bearing.z() * (Matrix3T::Identity() - bearing*bearing.transpose()) * Vector3T(opticalFlowNorm.x(), opticalFlowNorm.y(), 0);

    this->keypoint.pt = this->camCoordinates;

    this->pointColor = col;
    ++lifetime;
}
#include "Landmark.h"

using namespace GIFT;
using namespace cv;
using namespace Eigen;


Landmark::Landmark(const Point2f& newCamCoords, const Point2f& newCamCoordsNorm, int idNumber, const colorVec& col) {
    this->camCoordinates = newCamCoords;
    this->camCoordinatesNorm = newCamCoordsNorm;
    const double yNormInv = 1.0 / (newCamCoordsNorm.x*newCamCoordsNorm.x + newCamCoordsNorm.y*newCamCoordsNorm.y + 1.0);
    this->sphereCoordinates << yNormInv*newCamCoordsNorm.x, yNormInv*newCamCoordsNorm.y, yNormInv;

    this->opticalFlowRaw.setZero();
    this->opticalFlowNorm.setZero();

    this->pointColor = col;
    this->idNumber = idNumber;

    lifetime = 1;
}

void Landmark::update(const cv::Point2f& newCamCoords, const cv::Point2f& newCamCoordsNorm, const colorVec& col) {
    this->opticalFlowRaw << newCamCoords.x - this->camCoordinates.x, newCamCoords.y - this->camCoordinates.y;
    this->opticalFlowNorm << newCamCoordsNorm.x - this->camCoordinatesNorm.x, newCamCoordsNorm.y - this->camCoordinatesNorm.y;

    this->camCoordinates = newCamCoords;
    this->camCoordinatesNorm = newCamCoordsNorm;
    const double yNormInv = 1.0 / (newCamCoordsNorm.x*newCamCoordsNorm.x + newCamCoordsNorm.y*newCamCoordsNorm.y + 1.0);
    this->sphereCoordinates << yNormInv*newCamCoordsNorm.x, yNormInv*newCamCoordsNorm.y, yNormInv;
    ++lifetime;

    this->pointColor = col;
}
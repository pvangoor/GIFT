#include "Visualisation.h"

#include "opencv2/imgproc.hpp"
#include <algorithm>

using namespace GIFT;

Mat GIFT::drawFeatureImage(const Mat& baseImage, const vector<Landmark>& landmarks, const int& radius, const Scalar& color) {
    Mat featureImage = baseImage.clone();

    auto drawingLambda = [&](const Landmark& landmark) {
        circle(featureImage, landmark.camCoordinates, radius, color);
    };
    for_each(landmarks.begin(), landmarks.end(), drawingLambda);

    return featureImage;
}

Mat GIFT::drawFlowImage(const Mat& baseImage, const vector<Landmark>& landmarks0, const vector<Landmark>& landmarks1, const int& radius, const Scalar& circleColor, const int& thickness, const Scalar& lineColor){
    Mat featureImage = baseImage.clone();

    auto flowDrawingLambda = [&](const Landmark& lm0) {
        auto lm1it = find_if(landmarks1.begin(), landmarks1.end(), [&lm0](const Landmark& lm1) { return lm1.idNumber == lm0.idNumber; });
        if (lm1it != landmarks1.end()) {
            circle(featureImage, lm0.camCoordinates, radius, circleColor);
            line(featureImage, lm0.camCoordinates, lm1it->camCoordinates, lineColor, thickness);
        }
    };
    for_each(landmarks0.begin(), landmarks0.end(), flowDrawingLambda);

    return featureImage;
}
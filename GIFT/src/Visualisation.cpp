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
#include "iostream"
#include "string"
#include "vector"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "FeatureTracker.h"
#include "Configure.h"

int main(int argc, char *argv[]) {
    cv::String folderName = "/home/pieter/Documents/Datasets/ardupilot/flight1/";

    // Set up a monocular feature tracker
    GIFT::FeatureTracker ft = GIFT::FeatureTracker(GIFT::TrackerMode::MONO);
    GIFT::CameraParameters cam0 = GIFT::readCameraConfig(folderName+"/cam0.yaml");
    ft.setCameraConfiguration(cam0, 0);

    cv::VideoCapture cap(folderName+"small.mp4");
    cv::namedWindow("debug");

    cv::Mat image;

    while (cap.read(image)) {

        ft.processImage(image);

        std::vector<GIFT::Landmark> landmarks = ft.outputLandmarks();
        std::vector<cv::Point2f> features;
        for (const auto &lm : landmarks) {
            features.emplace_back(lm.camCoordinates[0]);
        }
        std::vector<cv::KeyPoint> keypoints;
        cv::KeyPoint::convert(features, keypoints);

        cv::Mat kpImage;
        cv::drawKeypoints(image, keypoints, kpImage, cv::Scalar(0,0,255));

        cv::imshow("debug", kpImage);
        cv::waitKey(1);
    }

    std::cout << "Complete." << std::endl;

}
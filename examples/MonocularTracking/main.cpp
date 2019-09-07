#include "iostream"
#include "string"
#include "vector"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "FeatureTracker.h"
#include "Configure.h"

int main(int argc, char *argv[]) {
    cv::String camConfigFile;
    cv::String videoFile;
    if (argc <= 1) {
        camConfigFile = "/home/pieter/Documents/Datasets/ardupilot/flight1/cam0.yaml";
        videoFile = "/home/pieter/Documents/Datasets/ardupilot/flight1/small.mp4";
    } else if (argc == 3) {
        camConfigFile = argv[1];
        videoFile = argv[2];        
    } else {
        throw std::runtime_error("You must provide exactly the camera calibration and the video file.");
    }

    cv::VideoCapture cap(videoFile);
    cv::namedWindow("debug");
    cv::Mat image;
    cap.read(image);

    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8U);
    mask(Rect(0, 0, 400, 300)) = 1;

    // Set up a monocular feature tracker
    GIFT::FeatureTracker ft = GIFT::FeatureTracker(GIFT::TrackerMode::MONO);
    GIFT::CameraParameters cam0 = GIFT::readCameraConfig(camConfigFile);
    ft.setCameraConfiguration(cam0, 0);
    ft.setMask(mask, 0);


    while (cap.read(image) ) {;

        ft.processImage(image);

        std::vector<GIFT::Landmark> landmarks = ft.outputLandmarks();
        std::vector<cv::Point2f> features;
        for (const auto &lm : landmarks) {
            features.emplace_back(lm.camCoordinates);
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
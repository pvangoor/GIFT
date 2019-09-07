#include "iostream"
#include "string"
#include "vector"
#include <stdexcept>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "FeatureTracker.h"
#include "Configure.h"
#include "EgoMotion.h"

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

    // Set up a monocular feature tracker
    GIFT::FeatureTracker ft = GIFT::FeatureTracker(GIFT::TrackerMode::MONO);
    GIFT::CameraParameters cam0 = GIFT::readCameraConfig(camConfigFile);
    ft.setCameraConfiguration(cam0, 0);
    ft.maxFeatures = 50;

    cv::VideoCapture cap(videoFile);
    cv::namedWindow("debug");

    cv::Mat image;
    int count = 0;
    int maxCount = 1000;

    while (cap.read(image) && ++count < maxCount) {

        // Track the features
        ft.processImage(image);
        std::vector<GIFT::Landmark> landmarks = ft.outputLandmarks();

        // Compute EgoMotion
        GIFT::EgoMotion egoMotion(landmarks);
        std::cout << "Estimated Linear Velocity:" << std::endl;
        std::cout << egoMotion.linearVelocity << '\n';
        std::cout << "Estimated Angular Velocity:" << std::endl;
        std::cout << egoMotion.angularVelocity << '\n' << std::endl;

        auto estFlows = egoMotion.estimateFlowsNorm(landmarks);


        // Draw the keypoints
        std::vector<cv::Point2f> features;
        for (const auto &lm : landmarks) {
            features.emplace_back(lm.camCoordinates);
        }
        std::vector<cv::KeyPoint> keypoints;
        cv::KeyPoint::convert(features, keypoints);
        cv::Mat kpImage;
        cv::drawKeypoints(image, keypoints, kpImage, cv::Scalar(0,0,255));
        cv::imshow("points", kpImage);

        // Draw the optical flow as well as the estimates
        cv::Mat flowImage;
        image.copyTo(flowImage);
        for (const auto &lm : landmarks) {
            cv::Point2f p1 = lm.camCoordinates;
            cv::Point2f p0 = p1 - cv::Point2f(lm.opticalFlowRaw.x(), lm.opticalFlowRaw.y());
            cv::line(flowImage, p0, p1, cv::Scalar(255,0,255));
            cv::circle(flowImage, p0, 2, cv::Scalar(255,0,0));
        }
        cv::imshow("flow", flowImage);

        // Draw the normalised flow and estimates
        constexpr int viewScale = 500;
        cv::Mat estFlowImage(viewScale*2, viewScale*2, CV_8UC3, Scalar(255,255,255));
        for (const auto& flow : estFlows) {
            cv::Point2f p1 = flow.first;
            cv::Point2f p0 = p1 - cv::Point2f(flow.second.x(), flow.second.y());
            p0 = cv::Point2f(viewScale,viewScale) + p0*viewScale;
            p1 = cv::Point2f(viewScale,viewScale) + p1*viewScale;
            cv::line(estFlowImage, p0, p1, cv::Scalar(255,0,0));
        }
        for (const auto &lm : landmarks) {
            cv::Point2f p1 = lm.camCoordinatesNorm;
            cv::Point2f p0 = p1 - cv::Point2f(lm.opticalFlowNorm.x(), lm.opticalFlowNorm.y());
            p0 = cv::Point2f(viewScale,viewScale) + p0*viewScale;
            p1 = cv::Point2f(viewScale,viewScale) + p1*viewScale;
            cv::line(estFlowImage, p0, p1, cv::Scalar(255,0,255));
            cv::circle(estFlowImage, p0, 2, cv::Scalar(255,0,0));
        }
        cv::imshow("estimated flow", estFlowImage);

        cv::waitKey(1);
    }

    std::cout << "Complete." << std::endl;

}
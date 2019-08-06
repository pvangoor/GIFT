#include "iostream"
#include "string"
#include "vector"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "FeatureTracker.h"
#include "Configure.h"

int main(int argc, char *argv[]) {
    cv::String folderName = "/home/pieter/Documents/Datasets/rectified/image_0";

    cv::Mat mask = cv::Mat::zeros(720,1280,CV_8U);
    mask(Rect(0, 60, 1250, 500)) = 1;

    // Set up a monocular feature tracker
    GIFT::FeatureTracker ft = GIFT::FeatureTracker(GIFT::TrackerMode::MONO);
    GIFT::CameraParameters cam0 = GIFT::readCameraConfig(folderName+"/cam0.yaml");
    ft.setCameraConfiguration(cam0, 0);
    ft.setMask(mask, 0);

    vector<cv::String> imageFileNames;
    cv::glob(folderName+"/*.png", imageFileNames);
    
    long int total = imageFileNames.size();
    int i = 0;
    cv::namedWindow("debug");
    cv::Mat image;

    for (const auto& fileName : imageFileNames) {
        image = cv::imread(fileName);

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


        std::cout << ++i << " / " << total << std::endl;
    }

    std::cout << "Complete." << std::endl;

}
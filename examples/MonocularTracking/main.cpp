#include "FeatureTracker.h"
#include "Configure.h"

#include "opencv2/highgui/highgui.hpp"

int main(int argc, char *argv[]) {

    GIFT::FeatureTracker ft(GIFT::readCameraConfig(cv::String(argv[1])));

    cv::VideoCapture cap;
    cap.open(cv::String(argv[2]));
    cv::Mat image;
    while (cap.read(image)) {;

        ft.processImage(image);
        std::vector<GIFT::Landmark> landmarks = ft.outputLandmarks();

        cv::Mat featureImage = ft.drawFeatureImage(Scalar(0,0,255), 5, 3);

        cv::imshow("debug", featureImage);
        int k = cv::waitKey(1);
        if (k == 27) break;
    }

}
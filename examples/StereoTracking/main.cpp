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

#include "StereoFeatureTracker.h"
#include "Configure.h"

#include "opencv2/highgui/highgui.hpp"

int main(int argc, char *argv[]) {

    GIFT::StereoFeatureTracker ft(GIFT::readCameraConfig(cv::String(argv[1])),
                                  GIFT::readCameraConfig(cv::String(argv[2])));


    cv::VideoCapture capLeft;
    capLeft.open(cv::String(argv[3]));
    cv::Mat imageLeft;
    cv::VideoCapture capRight;
    capRight.open(cv::String(argv[4]));
    cv::Mat imageRight;

    while (capLeft.read(imageLeft) && capRight.read(imageRight)) {;

        ft.processImages(imageLeft, imageRight);
        std::vector<GIFT::StereoLandmark> landmarks = ft.outputStereoLandmarks();

        // cv::Mat featureImage = ft.drawFeatureImage(Scalar(0,0,255), 5, 3);
        cv::Mat featureImageLeft = ft.drawStereoFeatureImage(Scalar(0,0,255), Scalar(0,255,255), 2, 2);

        cv::imshow("debug", featureImageLeft);
        int k = cv::waitKey(1);
        if (k == 27) break;
    }

}
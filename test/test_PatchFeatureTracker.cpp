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

#include "gtest/gtest.h"
#include "PatchFeatureTracker.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "ParameterGroup.h"

#include "Visualisation.h"

#include <fstream>

class PFTTest : public ::testing::Test {
protected:
    PFTTest() {
        img0 = imread(String(TEST_DATA_DIR) + String("img0.png"));
        img1 = imread(String(TEST_DATA_DIR) + String("img1.png"));
    }
    
public:
    Mat img0, img1;
    GIFT::PatchFeatureTracker<> pft;
};

TEST_F(PFTTest, DetectAndTrackLogic) {
    pft.settings.maximumFeatures = 20;
    pft.settings.minimumFeatureDistance = 20;
    pft.settings.minimumRelativeQuality = 0.01;
    pft.settings.patchSize = Size(9,9);
    pft.settings.pyramidLevels = 4;

    pft.detectFeatures(img0);
    vector<GIFT::Landmark> landmarks0 = pft.outputLandmarks();

    Point2f translationVec = Point2f(20,10);
    const Mat translationMat = (Mat_<double>(2,3) << 1, 0, translationVec.x, 0, 1, translationVec.y);
    Mat shiftedImg0; warpAffine(img0, shiftedImg0, translationMat, img0.size());

    pft.trackFeatures(shiftedImg0);
    vector<GIFT::Landmark> landmarks1 = pft.outputLandmarks();

    // Check basic logic
    ASSERT_EQ(landmarks0.size(), landmarks1.size());
    for (int i = 0; i < landmarks0.size(); ++i) {
        const GIFT::Landmark& lmi0 = landmarks0[i];
        const GIFT::Landmark& lmi1 = landmarks1[i];

        EXPECT_EQ(lmi0.idNumber, lmi1.idNumber);
        EXPECT_EQ(lmi0.lifetime, 0);
        EXPECT_EQ(lmi1.lifetime, 1);
    }

    // Check tracking success
    for (int i = 0; i < landmarks0.size(); ++i) {
        const GIFT::Landmark& lmi0 = landmarks0[i];
        const GIFT::Landmark& lmi1 = landmarks1[i];

        Point2f coordinateError = (lmi0.camCoordinates+translationVec - lmi1.camCoordinates);
        float coordinateErrorNorm = pow(coordinateError.dot(coordinateError), 0.5);
        EXPECT_LE(coordinateErrorNorm, 0.1);
    }

    // Mat flowImage = GIFT::drawFlowImage(img0, landmarks0, landmarks1);
    // imshow("Flow", flowImage);
    // waitKey(0);
}
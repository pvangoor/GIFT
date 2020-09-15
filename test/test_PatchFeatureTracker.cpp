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
    pft.detectFeatures(img0);
    vector<GIFT::Landmark> landmarks0 = pft.outputLandmarks();

    pft.trackFeatures(img1);
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


}
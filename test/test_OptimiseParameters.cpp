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
#include "OptimiseParameters.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "ParameterGroup.h"

class OptimiseParametersTest : public ::testing::Test {
protected:
    OptimiseParametersTest() {
        img0 = imread(dataDir + String("img0.png"));
        cvtColor(img0, img0, COLOR_BGR2GRAY);

        img1 = imread(dataDir + String("img1.png"));
        cvtColor(img1, img1, COLOR_BGR2GRAY);
        
    }
    
public:
    String dataDir = String(TEST_DATA_DIR);
    Mat img0, img1;
    
};

TEST_F(OptimiseParametersTest, AcceptsMinimum) {
    int numLevels = 1;
    ImageWithGradientPyramid pyr0 = ImageWithGradientPyramid(img0, numLevels);
    ImagePyramid pyr1 = ImagePyramid(img0, numLevels);

    Point2f basePoint = Point2f(250,350);
    PyramidPatch patch = extractPyramidPatch(basePoint, Size(21,21), pyr0);

    Affine2Group params = Affine2Group::Identity();
    optimiseParameters(params, patch, pyr1);

    ftype tfError = (params.transformation-Matrix2T::Identity()).norm();
    ftype tsError = (params.translation).norm();

    EXPECT_LE(tfError, 1e-3);
    EXPECT_LE(tsError, 1e-3);
}

TEST_F(OptimiseParametersTest, ConvergeSmallTranslationOnBase) {
    int numLevels = 1;
    ImageWithGradientPyramid pyr0 = ImageWithGradientPyramid(img0, numLevels);
    ImagePyramid pyr1 = ImagePyramid(img0, numLevels);

    Point2f basePoint = Point2f(250,350);
    PyramidPatch patch = extractPyramidPatch(basePoint, Size(21,21), pyr0);

    Affine2Group params = Affine2Group::Identity();

    // Create a slight offset (mistake)
    params.translation.x() = 1;


    optimiseParameters(params, patch, pyr1);

    ftype tfError = (params.transformation-Matrix2T::Identity()).norm();
    ftype tsError = (params.translation).norm();

    EXPECT_LE(tfError, 1e-3);
    EXPECT_LE(tsError, 1e-3);
}

TEST_F(OptimiseParametersTest, AcceptsMinimumInLevels) {
    int numLevels = 3;
    ImageWithGradientPyramid pyr0 = ImageWithGradientPyramid(img0, numLevels);
    ImagePyramid pyr1 = ImagePyramid(img0, numLevels);

    Point2f basePoint = Point2f(250,350);
    PyramidPatch patch = extractPyramidPatch(basePoint, Size(21,21), pyr0);

    Affine2Group params = Affine2Group::Identity();
    optimiseParameters(params, patch, pyr1);

    ftype tfError = (params.transformation-Matrix2T::Identity()).norm();
    ftype tsError = (params.translation).norm();

    EXPECT_LE(tfError, 1e-3);
    EXPECT_LE(tsError, 1e-3);
}
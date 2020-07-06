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

        Point2f basePoint = Point2f(350,300);
        int numLevels = 4;

        img0GradientPyrBase = ImageWithGradientPyramid(img0, 1);
        img0ImagePyrBase = ImagePyramid(img0, 1);
        img0PatchBase = extractPyramidPatch(basePoint, Size(21,21), img0GradientPyrBase);
        

        img0GradientPyrLevels = ImageWithGradientPyramid(img0, numLevels);
        img0ImagePyrLevels = ImagePyramid(img0, numLevels);
        img0PatchLevels = extractPyramidPatch(basePoint, Size(21,21), img0GradientPyrLevels);
    }
    
public:
    String dataDir = String(TEST_DATA_DIR);
    Mat img0, img1;
    ImageWithGradientPyramid img0GradientPyrBase, img0GradientPyrLevels;
    ImagePyramid img0ImagePyrBase, img0ImagePyrLevels;
    PyramidPatch img0PatchBase, img0PatchLevels;
};

TEST_F(OptimiseParametersTest, TranslationAcceptsMinimum) {
    TranslationGroup params = TranslationGroup::Identity();
    optimiseParameters(params, img0PatchBase, img0ImagePyrBase);

    ftype tsError = (params.translation).norm();
    EXPECT_LE(tsError, 1e-3);
}

TEST_F(OptimiseParametersTest, AffineAcceptsMinimum) {
    Affine2Group params = Affine2Group::Identity();
    optimiseParameters(params, img0PatchBase, img0ImagePyrBase);

    ftype tfError = (params.transformation-Matrix2T::Identity()).norm();
    ftype tsError = (params.translation).norm();

    EXPECT_LE(tfError, 1e-3);
    EXPECT_LE(tsError, 1e-3);
}

TEST_F(OptimiseParametersTest, TranslationConvergeSmallErrorOnBase) {
    for (int testIter=0; testIter<10; ++testIter) {
        TranslationGroup params;
        params.translation = Vector2T::Random()*3.0;

        optimiseParameters(params, img0PatchBase, img0ImagePyrBase);

        ftype tsError = (params.translation).norm();
        EXPECT_LE(tsError, 1e-2);
    }
}

TEST_F(OptimiseParametersTest, AffineConvergeSmallErrorOnBase) {
    for (int testIter=0; testIter<10; ++testIter) {
        Affine2Group params = Affine2Group::Identity();
        params.translation = Vector2T::Random()*3.0;

        optimiseParameters(params, img0PatchBase, img0ImagePyrBase);

        ftype tfError = (params.transformation-Matrix2T::Identity()).norm();
        ftype tsError = (params.translation).norm();

        EXPECT_LE(tfError, 1e-2);
        EXPECT_LE(tsError, 1e-2);
    }
}

TEST_F(OptimiseParametersTest, TranslationAcceptsMinimumInLevels) {
    TranslationGroup params = TranslationGroup::Identity();
    optimiseParameters(params, img0PatchLevels, img0ImagePyrLevels);

    ftype tsError = (params.translation).norm();
    EXPECT_LE(tsError, 1e-3);
}

TEST_F(OptimiseParametersTest, AffineAcceptsMinimumInLevels) {
    Affine2Group params = Affine2Group::Identity();
    optimiseParameters(params, img0PatchLevels, img0ImagePyrLevels);

    ftype tfError = (params.transformation-Matrix2T::Identity()).norm();
    ftype tsError = (params.translation).norm();
    EXPECT_LE(tfError, 1e-2);
    EXPECT_LE(tsError, 1e-2);
}

TEST_F(OptimiseParametersTest, TranslationConvergeErrorInLevels) {
    for (int testIter=0; testIter<10; ++testIter) {
        TranslationGroup params;
        params.translation = Vector2T::Random()*30.0;

        optimiseParameters(params, img0PatchLevels, img0ImagePyrLevels);

        ftype tsError = (params.translation).norm();
        EXPECT_LE(tsError, 1e-2);
    }
}

TEST_F(OptimiseParametersTest, AffineConvergeErrorInLevels) {
    for (int testIter=0; testIter<10; ++testIter) {
        Affine2Group params = Affine2Group::Identity();
        params.translation = Vector2T::Random()*30.0;

        optimiseParameters(params, img0PatchLevels, img0ImagePyrLevels);

        ftype tfError = (params.transformation-Matrix2T::Identity()).norm();
        ftype tsError = (params.translation).norm();
        EXPECT_LE(tfError, 1e-2);
        EXPECT_LE(tsError, 1e-2);
    }
}
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

#include "ImagePyramid.h"
#include "gtest/gtest.h"

TEST(ImagePyramidTest, PyrDimensions) {
    cv::Mat baseImage = cv::Mat::zeros(cv::Size(pow(2, 10), pow(2, 10)), CV_8UC1);
    ImagePyramid pyramid(baseImage, 5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(pyramid.levels[i].rows, baseImage.rows / pow(2, i));
        EXPECT_EQ(pyramid.levels[i].cols, baseImage.cols / pow(2, i));
    }
}
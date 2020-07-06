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

TEST(OptimiseParametersTest, nothing) {
    cv::String dataDir = cv::String(TEST_DATA_DIR);
    cv::Mat img0 = cv::imread(dataDir + cv::String("img0.png"));

    cv::imshow("test", img0);
    cv::waitKey(0);

    EXPECT_TRUE(true);
}
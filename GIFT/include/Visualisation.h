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

#pragma once

#include "Feature.h"
#include "opencv2/core.hpp"
#include <vector>

using namespace std;
using namespace cv;

namespace GIFT {

Mat drawFeatureImage(const Mat& baseImage, const vector<Feature>& landmarks, const int& radius = 3,
    const Scalar& color = Scalar(0, 255, 255));
Mat drawFlowImage(const Mat& baseImage, const vector<Feature>& landmarks0, const vector<Feature>& landmarks1,
    const int& radius = 3, const Scalar& circleColor = Scalar(0, 255, 255), const int& thickness = 2,
    const Scalar& lineColor = Scalar(255, 0, 255));
Mat drawFlowImage(const Mat& image0, const Mat& image1, const vector<Feature>& landmarks0,
    const vector<Feature>& landmarks1, const int& radius = 3, const Scalar& circleColor = Scalar(0, 255, 255),
    const int& thickness = 2, const Scalar& lineColor = Scalar(0, 255, 0));

} // namespace GIFT
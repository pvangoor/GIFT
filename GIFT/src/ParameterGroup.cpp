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

#include "ParameterGroup.h"

using namespace cv;

cv::Mat Affine2Group::actionJacobian(const cv::Point2f& point) const {
    float jacData[2][6] = { {point.x, point.y, 0, 0, 1, 0}, {0, 0, point.x, point.y, 1, 0} };
    Mat jac = - Mat(2, 6, CV_32FC1, &jacData);
    return jac;
}

void Affine2Group::applyStepLeft(const cv::Mat& step) {
    // Put the step in matrix form, exponentiate, and apply the result.
}

Affine2Group Affine2Group::Identity() {
    Affine2Group identityElement;
    identityElement.transformation = Mat2f::eye(2,2);
    identityElement.translation = Point2f(0,0);
    return identityElement;
}

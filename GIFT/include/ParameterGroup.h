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

#include "opencv2/core/core.hpp"

class ParameterGroup {
public:
    virtual cv::Mat actionJacobian(const cv::Point2f& point) const = 0;
    virtual void applyStepLeft(const cv::Mat& step) = 0;
};

class Affine2Group : public ParameterGroup {
public:
    cv::Mat2f transformation;
    cv::Point2f translation;

    cv::Mat actionJacobian(const cv::Point2f& point) const;
    void applyStepLeft(const cv::Mat& step);

    static Affine2Group Identity();
};
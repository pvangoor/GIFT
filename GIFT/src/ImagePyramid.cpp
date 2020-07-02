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
#include "opencv2/imgproc/imgproc.hpp"  

ImagePyramid::ImagePyramid(const cv::Mat& image, const int& numLevels) {
    assert(numLevels > 0);
    levels.resize(numLevels);
    levels[0] = image;
    for (int i=1; i<numLevels; ++i) {
        cv::pyrDown(levels[i-1], levels[i]);
    }
}

ImageWithGradient::ImageWithGradient(const cv::Mat& image) {
    this->image = image;
    cv::spatialGradient(image, this->gradientX, this->gradientY);
}

ImageWithGradientPyramid::ImageWithGradientPyramid(const cv::Mat& image, const int& numLevels) {
    assert(numLevels > 0);
    levels.resize(numLevels);
    levels[0] = ImageWithGradient(image);
    for (int i=1; i<numLevels; ++i) {
        cv::Mat temp;
        cv::pyrDown(levels[i-1].image, temp);
        levels[i] = ImageWithGradient(temp);
    }
}
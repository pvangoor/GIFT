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

#include <vector>
#include "opencv2/core/core.hpp"
#include "eigen3/Eigen/Core"
#include "ftype.h"

using namespace std;
using namespace Eigen;
using namespace cv;

struct ImagePyramid {
    vector<cv::Mat> levels;
    ImagePyramid(){};
    ImagePyramid(const cv::Mat& image, const int& numLevels);
};

struct ImageWithGradient {
    cv::Mat image;
    cv::Mat gradientX;
    cv::Mat gradientY;
    ImageWithGradient(){};
    ImageWithGradient(const cv::Mat& image);
};

struct ImageWithGradientPyramid {
    vector<ImageWithGradient> levels;
    ImageWithGradientPyramid(){};
    ImageWithGradientPyramid(const cv::Mat& image, const int& numLevels);
};

struct PyramidPatch {
    vector<VectorXT> vecImage;
    vector<Matrix<ftype, Dynamic, 2>> vecDifferential;
    Vector2T baseCentre;
    int rows;
    int cols;
};

struct ImagePatch {
    VectorXT vecImage;
    Matrix<ftype, Dynamic, 2> vecDifferential;
    Vector2T centre;
    int rows;
    int cols;
};

PyramidPatch extractPyramidPatch(const cv::Point2f& point, const cv::Size& sze, const ImageWithGradientPyramid& pyr);
vector<PyramidPatch> extractPyramidPatches(const vector<cv::Point2f>& points, const cv::Mat& image, const cv::Size& sze, const int& numLevels);
ImagePatch getPatchAtLevel(const PyramidPatch& pyrPatch, const int lv);
VectorXT vectoriseImage(const Mat& image);
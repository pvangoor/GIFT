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

#include "eigen3/Eigen/Core"
#include "ftype.h"
#include "opencv2/core/core.hpp"
#include <vector>

namespace GIFT {

struct ImagePyramid {
    std::vector<cv::Mat> levels;
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
    std::vector<ImageWithGradient> levels;
    ImageWithGradientPyramid(){};
    ImageWithGradientPyramid(const cv::Mat& image, const int& numLevels);
};

struct PyramidPatch {
    std::vector<Eigen::VectorXT> vecImage;
    std::vector<Eigen::Matrix<ftype, Eigen::Dynamic, 2>> vecDifferential;
    Eigen::Vector2T baseCentre;
    int rows;
    int cols;
    ftype at(int row, int col, int lv = 0) const;
};

struct ImagePatch {
    Eigen::VectorXT vecImage;
    Eigen::Matrix<ftype, Eigen::Dynamic, 2> vecDifferential;
    Eigen::Vector2T centre;
    int rows;
    int cols;
    ftype at(int row, int col) const;
};

PyramidPatch extractPyramidPatch(const cv::Point2f& point, const cv::Size& sze, const ImageWithGradientPyramid& pyr);
std::vector<PyramidPatch> extractPyramidPatches(
    const std::vector<cv::Point2f>& points, const cv::Mat& image, const cv::Size& sze, const int& numLevels);
ImagePatch getPatchAtLevel(const PyramidPatch& pyrPatch, const int lv);
Eigen::VectorXT vectoriseImage(const cv::Mat& image);

} // namespace GIFT
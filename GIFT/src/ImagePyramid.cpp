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

#include "GIFT/ImagePyramid.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <algorithm>

using namespace Eigen;
using namespace std;
using namespace cv;
using namespace GIFT;

ImagePyramid::ImagePyramid(const cv::Mat& image, const int& numLevels) {
    assert(numLevels > 0);
    levels.resize(numLevels);
    levels[0] = image;
    for (int i = 1; i < numLevels; ++i) {
        cv::pyrDown(levels[i - 1], levels[i]);
    }
}

ImageWithGradient::ImageWithGradient(const cv::Mat& image) {
    this->image = image;
    cv::Sobel(image, this->gradientX, CV_32F, 1, 0, -1, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(image, this->gradientY, CV_32F, 0, 1, -1, 1.0, 0.0, cv::BORDER_REPLICATE);
    this->gradientX = this->gradientX / 32.0;
    this->gradientY = this->gradientY / 32.0;
}

ImageWithGradientPyramid::ImageWithGradientPyramid(const cv::Mat& image, const int& numLevels) {
    assert(numLevels > 0);
    levels.resize(numLevels);
    levels[0] = ImageWithGradient(image);
    for (int i = 1; i < numLevels; ++i) {
        cv::Mat temp;
        cv::pyrDown(levels[i - 1].image, temp);
        levels[i] = ImageWithGradient(temp);
    }
}

ImageWithGradientPyramid::ImageWithGradientPyramid(const ImagePyramid& imagePyr) {
    levels.resize(imagePyr.levels.size());
    transform(imagePyr.levels.begin(), imagePyr.levels.end(), levels.begin(),
        [](const cv::Mat& image) { return ImageWithGradient(image); });
}

ftype ImagePatch::at(int row, int col) const {
    assert(row < rows());
    assert(col < cols());
    return pixelValue(imageWithGrad.image, row, col);
}

Eigen::Matrix<ftype, 1, 2> ImagePatch::differential(int row, int col) const {
    Eigen::Matrix<ftype, 1, 2> DI;
    DI << pixelValue(imageWithGrad.gradientX, row, col), pixelValue(imageWithGrad.gradientY, row, col);
    return DI;
}

Eigen::VectorXT ImagePatch::imageVector() const { return vectoriseImage(imageWithGrad.image); }

Eigen::Matrix<ftype, Eigen::Dynamic, 2> ImagePatch::imageVectorDifferential() const {
    Eigen::Matrix<ftype, Eigen::Dynamic, 2> DIVec(area(), 2);
    DIVec << vectoriseImage(imageWithGrad.gradientX), vectoriseImage(imageWithGrad.gradientY);
    return DIVec;
}

int PyramidPatch::totalPixelCount() const {
    int total = 0;
    for (const ImagePatch& level : levels) {
        total += level.area();
    }
    return total;
}

Eigen::VectorXT PyramidPatch::pyramidVector() const {
    Eigen::VectorXT PVec(totalPixelCount());
    int currentBase = 0;
    for (int lv = 0; lv < levels.size(); ++lv) {
        PVec.segment(currentBase, levels[lv].area()) = levels[lv].imageVector();
        currentBase += levels[lv].area();
    }

    return PVec;
}

Eigen::Matrix<ftype, Eigen::Dynamic, 2> PyramidPatch::pyramidVectorDifferential() const {
    Eigen::Matrix<ftype, Eigen::Dynamic, 2> DPVec(totalPixelCount(), 2);
    int currentBase = 0;
    for (int lv = 0; lv < levels.size(); ++lv) {
        // Note: the differential is halved each level up since the motion of a pixel on the base
        // corresponds to only half that motion on the level above.
        DPVec.block(currentBase, 0, levels[lv].area(), 2) = levels[lv].imageVectorDifferential() * pow(lv, -2);
        currentBase += levels[lv].area();
    }

    return DPVec;
}

ftype PyramidPatch::at(int row, int col, int lv) const {
    assert(lv < levels.size());
    assert(row < rows(lv));
    assert(col < cols(lv));
    return pixelValue(levels[lv].imageWithGrad.image, row, col);
}

ImagePatch GIFT::extractImagePatch(
    const cv::Point2f& point, const cv::Size& sze, const ImageWithGradient& imageWithGrad) {
    ImagePatch patch;
    patch.centre = Vector2T(point.x, point.y);
    getRectSubPix(imageWithGrad.image, sze, point, patch.imageWithGrad.image, CV_32F);
    getRectSubPix(imageWithGrad.gradientX, sze, point, patch.imageWithGrad.gradientX, CV_32F);
    getRectSubPix(imageWithGrad.gradientY, sze, point, patch.imageWithGrad.gradientY, CV_32F);
    return patch;
}

PyramidPatch GIFT::extractPyramidPatch(
    const cv::Point2f& point, const cv::Size& sze, const ImageWithGradientPyramid& pyr) {
    int numLevels = pyr.levels.size();
    PyramidPatch patch;
    patch.levels.resize(numLevels);
    for (int lv = 0; lv < numLevels; ++lv) {
        patch.levels[lv] = extractImagePatch(point * pow(2, -lv), sze, pyr.levels[lv]);
    }
    return patch;
}

PyramidPatch extractPyramidPatch(
    const cv::Point2f& point, const std::vector<cv::Size>& sizes, const ImageWithGradientPyramid& pyr) {
    int numLevels = pyr.levels.size();
    assert(numLevels == sizes.size());
    PyramidPatch patch;
    patch.levels.resize(numLevels);
    for (int lv = 0; lv < numLevels; ++lv) {
        patch.levels[lv] = extractImagePatch(point * pow(2, -lv), sizes[lv], pyr.levels[lv]);
    }
    return patch;
}

vector<GIFT::PyramidPatch> GIFT::extractPyramidPatches(
    const vector<cv::Point2f>& points, const cv::Mat& image, const cv::Size& sze, const int& numLevels) {
    ImageWithGradientPyramid pyr(image, numLevels);
    auto patchLambda = [pyr, sze](const cv::Point2f& point) { return extractPyramidPatch(point, sze, pyr); };
    vector<PyramidPatch> patches(points.size());
    transform(points.begin(), points.end(), patches.begin(), patchLambda);
    return patches;
}

VectorXT GIFT::vectoriseImage(const Mat& image) {
    // We work row by row
    const int rows = image.rows;
    const int cols = image.cols;
    VectorXT vecImage(rows * cols);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            vecImage(x + y * rows) = pixelValue(image, y, x);
        }
    }
    return vecImage;
}

ftype GIFT::pixelValue(const Mat& image, const int& row, const int& col) {
    // We work row by row
    switch (image.depth()) {
    case CV_8U:
        return (ftype)image.at<uchar>(row, col);
        break;
    case CV_16S:
        return (ftype)image.at<short>(row, col);
        break;
    case CV_32F:
        return (ftype)image.at<float>(row, col);
        break;
    case CV_64F:
        return (ftype)image.at<double>(row, col);
        break;
    }
    return nan("");
}
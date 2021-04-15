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

#include "GIFT/OptimiseParameters.h"
#include "GIFT/ParameterGroup.h"
#include "eigen3/Eigen/Dense"

using namespace Eigen;
using namespace std;
using namespace cv;

int clamp(const int x, const int a, const int b) {
    if (x < a)
        return a;
    if (x > b)
        return b;
    return x;
}

void optimiseParametersAtLevel(ParameterGroup& params, const ImagePatch& patch, const Mat& image);
MatrixXT patchActionJacobian(const ParameterGroup& params, const ImagePatch& patch);
VectorXT paramResidual(const ParameterGroup& params, const ImagePatch& patch, const Mat& image);

void optimiseParameters(vector<ParameterGroup>& params, const vector<PyramidPatch>& patches, const Mat& image) {
    if (patches.size() == 0)
        return;
    optimiseParameters(params, patches, ImagePyramid(image, patches[0].vecImage.size()));
}

void optimiseParameters(
    vector<ParameterGroup>& params, const vector<PyramidPatch>& patches, const ImagePyramid& pyramid) {
    assert(patches.size() == params.size());
    for (int i = 0; i < params.size(); ++i) {
        optimiseParameters(params[i], patches[i], pyramid);
    }
}

void optimiseParameters(ParameterGroup& params, const PyramidPatch& patch, const ImagePyramid& pyramid) {
    const int numLevels = patch.vecImage.size();
    for (int lv = numLevels - 1; lv >= 0; --lv) {
        params.changeLevel(lv);
        optimiseParametersAtLevel(params, getPatchAtLevel(patch, lv), pyramid.levels[lv]);
    }
    params.changeLevel(0);
}

void optimiseParametersAtLevel(ParameterGroup& params, const ImagePatch& patch, const Mat& image) {
    // Use the Inverse compositional algorithm to optimise params at the given level.
    const MatrixXT jacobian = patchActionJacobian(params, patch);
    const MatrixXT stepOperator = (jacobian.transpose() * jacobian).inverse() * jacobian.transpose();
    VectorXT previousStepDirection = VectorXT::Zero(params.dim());

    for (int iteration = 0; iteration < 50; ++iteration) {
        MatrixXT residualVector = paramResidual(params, patch, image);
        MatrixXT stepVector = -stepOperator * residualVector;
        if (stepVector.norm() < 1e-3)
            break;

        VectorXT stepDirection = stepVector.normalized();
        if (stepDirection.dot(previousStepDirection) < -0.5)
            stepVector = stepVector / 2.0;
        previousStepDirection = stepDirection;

        params.applyStepOnRight(stepVector);
    }
}

MatrixXT patchActionJacobian(const ParameterGroup& params, const ImagePatch& patch) {
    // The patch is vectorised row by row.
    const Vector2T offset = 0.5 * Vector2T(patch.cols - 1, patch.rows - 1);
    MatrixXT jacobian(patch.rows * patch.cols, params.dim());
    for (int y = 0; y < patch.rows; ++y) {
        for (int x = 0; x < patch.cols; ++x) {
            const Vector2T point = Vector2T(x, y) - offset;
            int rowIdx = (x + y * patch.rows);

            jacobian.block(rowIdx, 0, 1, params.dim()) =
                patch.vecDifferential.block<1, 2>(rowIdx, 0) * params.actionJacobian(point);
        }
    }
    return jacobian;
}

VectorXT paramResidual(const ParameterGroup& params, const ImagePatch& patch, const Mat& image) {
    const Vector2T offset = 0.5 * Vector2T(patch.rows - 1, patch.cols - 1);
    VectorXT residualVector = VectorXT(patch.rows * patch.cols);
    for (int y = 0; y < patch.rows; ++y) {
        for (int x = 0; x < patch.cols; ++x) {
            const Vector2T point = Vector2T(x, y) - offset;
            const Vector2T transformedPoint = patch.centre + params.applyLeftAction(point);
            const float subPixelValue = getSubPixel(image, transformedPoint);
            residualVector(x + y * patch.rows) = subPixelValue - patch.vecImage(x + y * patch.rows);
        }
    }
    return residualVector;
}

float getSubPixel(const Mat& image, const Vector2T& point) {
    // Replicate the border outside the image
    // const int x0 = clamp((int)point.x(), 0, image.cols-2);
    // const int y0 = clamp((int)point.y(), 0, image.rows-2);
    int x0 = (int)point.x();
    int y0 = (int)point.y();
    const float dx = (x0 >= 0 && x0 < image.cols - 1) ? (point.x() - x0) : 0.0;
    const float dy = (y0 >= 0 && y0 < image.rows - 1) ? (point.y() - y0) : 0.0;
    x0 = clamp(x0, 0, image.cols - 1);
    y0 = clamp(y0, 0, image.rows - 1);
    const uchar im00 = image.at<uchar>(y0, x0);
    const uchar im01 = image.at<uchar>(y0 + 1, x0);
    const uchar im10 = image.at<uchar>(y0, x0 + 1);
    const uchar im11 = image.at<uchar>(y0 + 1, x0 + 1);

    const float value =
        dx * dy * im11 + dx * (1.0 - dy) * im10 + (1.0 - dx) * dy * im01 + (1.0 - dx) * (1.0 - dy) * im00;
    return value;
}
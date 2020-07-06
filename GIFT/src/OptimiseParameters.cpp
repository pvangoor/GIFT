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

#include "OptimiseParameters.h"
#include "ParameterGroup.h"

using namespace Eigen;
using namespace cv;

void optimiseParametersAtLevel(ParameterGroup& params, const ImagePatch& patch, const Mat& image);
MatrixXT patchActionJacobian(const ParameterGroup& params, const ImagePatch& patch);
VectorXT paramResidual(const ParameterGroup& params, const ImagePatch& patch, const Mat& image);
float getSubPixel(const Mat& image, const Vector2T& point);

void optimiseParameters(ParameterGroup& params, const PyramidPatch& patch, const ImagePyramid& pyramid) {
    const int numLevels = patch.vecImage.size();
    for (int lv=numLevels-1; lv>=0; --lv) {
        optimiseParametersAtLevel(params, getPatchAtLevel(patch, lv), pyramid.levels[lv]);
    }
}

void optimiseParametersAtLevel(ParameterGroup& params, const ImagePatch& patch, const Mat& image) {
    // Use the Inverse compositional algorithm to optimise params at the given level.
    MatrixXT jacobian = - patch.vecDifferential * params.actionJacobian(patch.centre);
    MatrixXT stepOperator = (jacobian.transpose() * jacobian).inverse() * jacobian.transpose();

    for (int iteration = 0; iteration < 30; ++iteration) {
        MatrixXT residualVector = paramResidual(params, patch, image);
        MatrixXT stepVector = stepOperator * residualVector;
        params.applyStepLeft(stepVector);
        if (stepVector.norm() < 1e-6) break;
    }
}

MatrixXT patchActionJacobian(const ParameterGroup& params, const ImagePatch& patch) {
    // Vectorise the patch row by row
    const Vector2T offset = 0.5 * Vector2T(patch.cols, patch.rows);
    MatrixXT jacobian(2*patch.rows*patch.cols, params.dim());
    for (int y=0; y<patch.rows; ++y) {
        for (int x=0; x<patch.cols; ++x) {
            const Vector2T point = Vector2T(x,y) - offset;
            int rowIdx = (x+y*patch.rows);
            jacobian.block(rowIdx, 0, 2, params.dim()) = params.actionJacobian(point);
        }
    }
    return jacobian;
}

VectorXT paramResidual(const ParameterGroup& params, const ImagePatch& patch, const Mat& image) {
    const Vector2T offset = 0.5 * Vector2T(patch.rows, patch.cols);
    VectorXT residualVector = VectorXT(patch.rows*patch.cols);
    for (int y=0; y<patch.rows; ++y) {
        for (int x=0; x<patch.cols; ++x) {
            const Vector2T point = Vector2T(x,y) - offset;
            const Vector2T transformedPoint = params.applyAction(point);
            const float subPixelValue = getSubPixel(image, patch.centre+transformedPoint);
            residualVector(x+y*patch.rows) = patch.vecImage(x+y*patch.rows) - subPixelValue;
        }
    }
    return residualVector;
}

float getSubPixel(const Mat& image, const Vector2T& point) {
    const int x0 = (int)point.x();
    const int y0 = (int)point.y();
    const float dx = point.x() - x0;
    const float dy = point.y() - y0;
    const float value = dx*dy*image.at<uchar>(y0,x0) + dx*(1.0-dy)*image.at<uchar>(y0+1,x0)
                        + (1.0-dx)*dy*image.at<uchar>(y0,x0+1) + (1.0-dx)*(1.0-dy)*image.at<uchar>(y0+1,x0+1);
    return value;
}
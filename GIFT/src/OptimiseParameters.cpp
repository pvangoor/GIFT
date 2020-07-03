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

using namespace cv;

void optimiseParametersAtLevel(ParameterGroup& params, const ImagePatch& patch, const Mat& image);
Mat patchJacobian(const ImagePatch& patch);
Mat paramResidual(const ParameterGroup& params, const ImagePatch& patch, const Mat& image);

void optimiseParameters(ParameterGroup& params, const PyramidPatch& patch, const ImagePyramid& pyramid) {
    const int numLevels = patch.patch.levels.size();
    for (int lv=numLevels-1; lv>=0; --lv) {
        optimiseParametersAtLevel(params, getPatchAtLevel(patch, lv), pyramid.levels[lv]);
    }
}

void optimiseParametersAtLevel(ParameterGroup& params, const ImagePatch& patch, const Mat& image) {
    // Use the Inverse compositional algorithm to optimise params at the given level.
    Mat jacobian = - patchJacobian(patch) * params.actionJacobian(patch.point);
    Mat stepOperator = (jacobian.t() * jacobian).inv() * jacobian.t();

    for (int iteration = 0; iteration < 30; ++iteration) {
        Mat residualVector = paramResidual(params, patch, image);
        Mat stepVector = stepOperator * residualVector;
        params.applyStepLeft(stepVector);
        if (norm(stepVector) < 1e-6) break;
    }
}

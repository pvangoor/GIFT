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

using namespace cv;

void optimiseParametersAtLevel(ParameterGroup& params, const ImagePatch& patch, const cv::Mat& image);

void optimiseParameters(ParameterGroup& params, const PyramidPatch& patch, const ImagePyramid& pyramid) {
    const int numLevels = patch.patch.levels.size();
    for (int lv=numLevels-1; lv>=0; --lv) {
        optimiseParametersAtLevel(params, getPatchAtLevel(patch, lv), pyramid.levels[lv]);
    }
}

void optimiseParametersAtLevel(ParameterGroup& params, const ImagePatch& patch, const cv::Mat& image) {
    // Optimise the transformation parameters using the inverse compositional algorithm.
}
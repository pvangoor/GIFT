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

#include "GIFeatureTracker.h"
#include "ImagePyramid.h"
#include "ParameterGroup.h"

namespace GIFT {

class PatchFeatureTracker : public GIFeatureTracker {
protected:
    // Transform parameters and patches
    struct InternalPatchFeature {
        ImagePatch patch;
        TranslationGroup parameters;
    };
    vector<InternalPatchFeature> features;

public:
    // Initialisation and configuration
    PatchFeatureTracker(const CameraParameters& cameraParams);
    PatchFeatureTracker(const CameraParameters& cameraParams, const Mat& mask);

    // Core
    virtual void detectFeatures(const Mat &image) override {};
    virtual void trackFeatures(const Mat &image) override {};
    [[nodiscard]] virtual vector<Landmark> outputLandmarks() const override {return {}; };
};
}


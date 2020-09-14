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
#include "OptimiseParameters.h"

#include "opencv2/imgproc.hpp"

namespace GIFT {

template<class PG = TranslationGroup>
class PatchFeatureTracker : public GIFeatureTracker {
protected:
    // Transform parameters and patches
    static_assert(std::is_base_of<ParameterGroup, PG>::value, "Patch tracker template is not derived from ParameterGroup.");
    struct InternalPatchFeature {
        PyramidPatch patch;
        PG parameters;
        int id = -1;
        int lifetime = 0;
    };
    vector<InternalPatchFeature> features;

    // Settings
    int maximumFeatures = 50;
    double minimumFeatureDistance = 10;
    double minimumRelativeQuality = 0.05;
    int pyramidLevels = 3;
    Size patchSize = Size(21,21);

public:
    // Initialisation and configuration
    PatchFeatureTracker(const CameraParameters& cameraParams) : GIFeatureTracker(cameraParams) {};
    PatchFeatureTracker(const CameraParameters& cameraParams, const Mat& mask) : GIFeatureTracker(cameraParams, mask) {};

    // Core
    virtual void detectFeatures(const Mat &image) override {
        vector<Point2f> points;
        goodFeaturesToTrack(image, points, maximumFeatures, minimumRelativeQuality, minimumFeatureDistance);
        vector<PyramidPatch> patches = extractPyramidPatches(points, image, patchSize, pyramidLevels);

        auto featureLambda = [this](const PyramidPatch& patch) { 
            InternalPatchFeature feature;
            feature.patch = patch;
            feature.parameters = PG::Identity();
            feature.id = ++this->currentNumber;
            return feature;
        };

        features.resize(patches.size());
        transform(patches.begin(), patches.end(), features.begin(), featureLambda);
        // TODO: This resets the features every time you call it. What would be better?
        // Probably it should refresh the existing features and add new features up to a maximum number.
    };

    virtual void trackFeatures(const Mat &image) override {
        ImagePyramid newPyr(image, pyramidLevels);
        for_each(features.begin(), features.end(), [&newPyr](InternalPatchFeature& feature) {
            optimiseParameters(feature.parameters, feature.patch, newPyr);
            ++feature.lifetime;
        });
        // TODO: We need to remove features that are no longer visible.
    };

    [[nodiscard]] virtual vector<Landmark> outputLandmarks() const override {
        // TODO: Convert all the patch features to landmarks
        vector<Landmark> landmarks(features.size());
        transform(features.begin(), features.end(), landmarks.begin(), [this](const InternalPatchFeature& f){
            return this->featureToLandmark(f);
        });
        return landmarks;
    };

    [[nodiscard]] Landmark featureToLandmark(const InternalPatchFeature& feature) const {
        Landmark lm;
        const Vector2T normalCamCoords = feature.parameters.applyLeftAction(feature.patch.baseCentre);
        lm.camCoordinatesNorm = Point2f(normalCamCoords.x(), normalCamCoords.y());
        lm.idNumber = feature.id;
        lm.lifetime = feature.lifetime;
        lm.pointColor.fill(feature.patch.at(feature.patch.rows/2, feature.patch.cols/2));
        return lm;
    }
};
}


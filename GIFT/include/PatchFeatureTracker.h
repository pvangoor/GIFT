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
#include "opencv2/calib3d.hpp"

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
        Point2f camCoordinates() const {
            const Vector2T result = parameters.applyLeftAction(patch.baseCentre);
            return Point2f(result.x(), result.y());
        }
    };

    // Feature storage
    vector<InternalPatchFeature> features;

    // Settings
    int maximumFeatures = 20;
    double minimumFeatureDistance = 20;
    double minimumRelativeQuality = 0.05;
    int pyramidLevels = 3;
    Size patchSize = Size(21,21);

public:
    // Initialisation and configuration
    PatchFeatureTracker() = default;
    PatchFeatureTracker(const CameraParameters& cameraParams) : GIFeatureTracker(cameraParams) {};
    PatchFeatureTracker(const CameraParameters& cameraParams, const Mat& mask) : GIFeatureTracker(cameraParams, mask) {};

    // Core
    virtual void detectFeatures(const Mat &image) override {
        // Detect new points
        vector<Point2f> newPoints;
        Mat gray = image;
        if (gray.channels() > 1) cvtColor(image, gray, COLOR_RGB2GRAY);
        goodFeaturesToTrack(gray, newPoints, maximumFeatures, minimumRelativeQuality, minimumFeatureDistance);
        
        // Remove new points that are too close to existing features
        vector<Point2f> oldPoints(features.size());
        transform(features.begin(), features.end(), oldPoints.begin(), [](const InternalPatchFeature& f) {return f.camCoordinates();});
        removePointsTooClose(newPoints, oldPoints, minimumFeatureDistance);
        const int numPointsToAdd = maximumFeatures - oldPoints.size();
        newPoints.resize(max(numPointsToAdd, 0));

        // Convert the new points to patch features
        vector<PyramidPatch> newPatches = extractPyramidPatches(newPoints, gray, patchSize, pyramidLevels);
        auto newFeatureLambda = [this](const PyramidPatch& patch) { 
            InternalPatchFeature feature;
            feature.patch = patch;
            feature.parameters = PG::Identity();
            feature.id = ++this->currentNumber;
            return feature;
        };
        vector<InternalPatchFeature> newFeatures(newPatches.size());
        transform(newPatches.begin(), newPatches.end(), newFeatures.begin(), newFeatureLambda);

        // Append
        features.insert(features.end(), newFeatures.begin(), newFeatures.end());

        // TODO: This only adds new features. Ideally, we also refresh existing features.
    };

    virtual void trackFeatures(const Mat &image) override {
        Mat gray = image; if (gray.channels() > 1) cvtColor(image, gray, COLOR_RGB2GRAY);
        ImagePyramid newPyr(gray, pyramidLevels);
        for_each(features.begin(), features.end(), [&newPyr](InternalPatchFeature& feature) {
            optimiseParameters(feature.parameters, feature.patch, newPyr);
            ++feature.lifetime;
        });
        // TODO: We need to remove features that are no longer visible.
    };

    [[nodiscard]] virtual vector<Landmark> outputLandmarks() const override {
        vector<Landmark> landmarks(features.size());
        transform(features.begin(), features.end(), landmarks.begin(), [this](const InternalPatchFeature& f){
            return this->featureToLandmark(f);
        });
        return landmarks;
    };

    [[nodiscard]] Landmark featureToLandmark(const InternalPatchFeature& feature) const {
        Landmark lm;
        lm.camCoordinates = feature.camCoordinates();
        // undistortPoints(lm.camCoordinates, lm.camCoordinatesNorm, camera.K, camera.distortionParams);
        lm.idNumber = feature.id;
        lm.lifetime = feature.lifetime;
        lm.pointColor.fill(feature.patch.at(feature.patch.rows/2, feature.patch.cols/2));
        return lm;
        // TODO: Some parts of the landmark are missing. Is this a problem?
    }

    static void removePointsTooClose(vector<Point2f> newPoints, const vector<Point2f>& oldPoints, const double& minDist) {
        const double minDistSq = minDist*minDist;
        for (int i=newPoints.size()-1; i>=0; --i) {
            Point2f& newPoint = newPoints[i];
            for (const Point2f& point : oldPoints) {
                const double distSq = (point.x - newPoint.x)*(point.x - newPoint.x) + (point.y - newPoint.y)*(point.y - newPoint.y);
                if (distSq < minDistSq) {
                    newPoints.erase(newPoints.begin()+i);
                    break;
                }
            }
        }
    }
};
}


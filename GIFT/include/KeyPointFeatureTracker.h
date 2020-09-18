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

#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"

namespace GIFT {

class KeyPointFeatureTracker : public GIFeatureTracker {
protected:
    // Transform parameters and patches
    struct InternalKPFeature {
        KeyPoint kp;
        int id = -1;
        int lifetime = 0;
        Point2f camCoordinates() const {
            return kp.pt;
        }
    };

    vector<InternalKPFeature> features; // Feature storage

public:
    // Settings
    struct Settings {
        int maximumFeatures = 20;
        // double minimumFeatureDistance = 20;
        // double minimumRelativeQuality = 0.05;
    };
    Settings settings;

    // Initialisation and configuration
    KeyPointFeatureTracker() = default;
    KeyPointFeatureTracker(const CameraParameters& cameraParams) : GIFeatureTracker(cameraParams) {};
    KeyPointFeatureTracker(const CameraParameters& cameraParams, const Mat& mask) : GIFeatureTracker(cameraParams, mask) {};

    // Core
    virtual void detectFeatures(const Mat &image) override;

    virtual void trackFeatures(const Mat &image) override;

    [[nodiscard]] virtual vector<Landmark> outputLandmarks() const override;

    [[nodiscard]] Landmark featureToLandmark(const InternalKPFeature& feature) const;
};

} // namespace GIFT


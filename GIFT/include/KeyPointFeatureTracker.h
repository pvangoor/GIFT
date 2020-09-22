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

#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"

namespace GIFT {

class KeyPointFeatureTracker : public GIFeatureTracker {
  protected:
    struct InternalKPFeature {
        KeyPoint kp;
        Mat descriptor;
        int id = -1;
        int lifetime = 0;
        double descriptorDist = 0;
        Point2f camCoordinates() const { return kp.pt; }
    };

    vector<InternalKPFeature> features; // Feature storage
    Ptr<ORB> ORBDetector = ORB::create();
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING);

  public:
    // Settings
    struct Settings {
        int maximumFeatures = 20;
        double minimumFeatureDistance = 20;
    };
    Settings settings; // TODO expand these settings to actually change the detector

    // Initialisation and configuration
    KeyPointFeatureTracker() = default;
    KeyPointFeatureTracker(const Camera& cameraParams) : GIFeatureTracker(cameraParams){};
    KeyPointFeatureTracker(const Camera& cameraParams, const Mat& mask) : GIFeatureTracker(cameraParams, mask){};

    // Core
    virtual void detectFeatures(const Mat& image) override;

    virtual void trackFeatures(const Mat& image) override;

    [[nodiscard]] virtual vector<Landmark> outputLandmarks() const override;

    [[nodiscard]] Landmark featureToLandmark(const InternalKPFeature& feature) const;

    void removePointsTooCloseToFeatures(vector<InternalKPFeature>& newKeypoints) const;
    static void filterForBestPoints(
        vector<InternalKPFeature>& proposedFeatures, const int& maxFeatures, const double& minDist);
};

} // namespace GIFT

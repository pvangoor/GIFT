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

#include <vector>

#include "Camera.h"
#include "Feature.h"
#include "opencv2/core.hpp"

namespace GIFT {

class GIFeatureTracker {
  protected:
    int currentNumber = 0;
    std::shared_ptr<Camera> cameraPtr;
    cv::Mat mask;

  public:
    // Initialisation and configuration
    GIFeatureTracker(){};
    GIFeatureTracker(const Camera& cameraParams);
    GIFeatureTracker(const Camera& cameraParams, const cv::Mat& mask);
    virtual ~GIFeatureTracker(){};
    virtual void setCamera(const Camera& cameraParameters);
    virtual void setMask(const cv::Mat& mask);

    // Core
    virtual void detectFeatures(const cv::Mat& image) = 0;
    virtual void trackFeatures(const cv::Mat& image) = 0;
    virtual std::vector<Feature> outputFeatures() const = 0;
};
} // namespace GIFT

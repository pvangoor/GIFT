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

#include <random>
#include <vector>

#include <GIFT/Feature.h>

namespace GIFT {

struct RansacParameters {
    size_t minDataPoints = 10;
    size_t maxIterations = 5;
    ftype inlierThreshold = 0.1;
    size_t minInliers = 20;
};

std::vector<GIFT::Feature> determineStaticWorldInliers(
    std::vector<GIFT::Feature>& features, const RansacParameters& params);

Eigen::Matrix3T fitEssentialMatrix(const std::vector<GIFT::Feature>& features);

template <typename T>
std::vector<T> sampleVector(const std::vector<T>& items, const size_t& n, std::mt19937& generator) {
    // This is basic reservoir sampling
    std::vector<T> sample(n);
    sample.insert(sample.begin(), items.begin(), items.begin() + n);

    for (size_t i = n; i < items.size(); ++i) {
        std::uniform_int_distribution<size_t> distribution(0, i);
        const std::size_t j = distribution(generator);
        if (j < sample.size()) {
            sample[j] = items[i];
        }
    }

    return sample;
}

}; // namespace GIFT
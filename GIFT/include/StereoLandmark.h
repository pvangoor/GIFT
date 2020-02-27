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
#include <array>
#include "eigen3/Eigen/Dense"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "Landmark.h"

using colorVec = std::array<uchar, 3>;

namespace GIFT {

struct StereoLandmark {
    GIFT::Landmark landmarkLeft;
    GIFT::Landmark landmarkRight;
 
    int idNumberStereo;
    int lifetime = 0;

    StereoLandmark();
    StereoLandmark(GIFT::Landmark lmLeft) {
        this->landmarkLeft = lmLeft;
        this->idNumberStereo = lmLeft.idNumber;
    }
    StereoLandmark(GIFT::Landmark lmLeft, GIFT::Landmark lmRight) {
        this->landmarkLeft = lmLeft;
        this->landmarkRight = lmRight;
        this->idNumberStereo = lmLeft.idNumber;
    }
    // void update();

};

}
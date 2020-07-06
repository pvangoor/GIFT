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

#include "ParameterGroup.h"

Matrix<ftype, 2, Dynamic> Affine2Group::actionJacobian(const Vector2T& point) const {
    Matrix<ftype, 2, Dynamic> jac;
    jac << point.x(), point.y(), 0, 0, 1, 0, 
           0, 0, point.x(), point.y(), 0, 1;
    return - jac;
}

void Affine2Group::applyStepLeft(const VectorXT& step) {
    // Put the step in matrix form, exponentiate, and apply the result.
}

Affine2Group Affine2Group::Identity() {
    Affine2Group identityElement;
    identityElement.transformation = Matrix2T::Identity();
    identityElement.translation = Vector2T::Zero();
    return identityElement;
}

Vector2T Affine2Group::applyAction(const Vector2T& point) const {
    Vector2T transformedPoint = this->transformation * point + this->translation;
    return transformedPoint;
}
// Copyright 2023 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <sstream>
#include <algorithm>
#include <cassert>

extern std::string
format(float f, int digits);

extern float
clamp(const float val, const float minVal, const float maxVal);



inline float dequantInt8(uint8_t qv, const float qp_scale, const float qp_zp)  
{
  return ((float)qv - qp_zp) * qp_scale;
}

inline float dequantInt16(uint16_t qv, const float qp_scale, const float qp_zp)
{
  return ((float)qv - qp_zp) * qp_scale;
}

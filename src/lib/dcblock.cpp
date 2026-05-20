/*
Copyright (c) 2019-2023 Jakob Ketterl <jakob.ketterl@gmx.de>

This file is part of libcsdr.

libcsdr is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

libcsdr is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with libcsdr.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "dcblock.hpp"

#include <algorithm>
#include <cmath>

using namespace Csdr;

DcBlock::DcBlock(float sampleRate, float cutoff, float fadeTime):
    estimateAlpha(alphaFor(sampleRate, cutoff)),
    fadeAlpha(alphaFor(sampleRate, 1.0f / clampPositive(fadeTime, 0.05f)))
{}

float DcBlock::clampPositive(float value, float fallback) {
    if (!std::isfinite(value) || value <= 0.0f) return fallback;
    return value;
}

float DcBlock::alphaFor(float sampleRate, float hz) {
    sampleRate = clampPositive(sampleRate, 48000.0f);
    hz = clampPositive(hz, 15.0f);
    float normalized = 1.0f - std::exp(-2.0f * static_cast<float>(M_PI) * hz / sampleRate);
    return std::min(std::max(normalized, 0.0f), 1.0f);
}

void DcBlock::process(float *input, float *output, size_t length) {
    for (size_t i = 0; i < length; i++) {
        float x = input[i];
        if (std::isnan(x)) x = 0.0f;

        // Track the underlying offset with a sample-rate-aware low-pass, then
        // fade the applied correction to avoid abrupt state changes.
        dcEstimate += estimateAlpha * (x - dcEstimate);
        appliedDc += fadeAlpha * (dcEstimate - appliedDc);
        output[i] = x - appliedDc;
    }
}

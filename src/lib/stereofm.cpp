/*
Copyright (c) 2026

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

#include "stereofm.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace Csdr;

namespace {
    constexpr float pilotFrequency = 19000.0f;
    constexpr float stereoFrequency = 38000.0f;
    constexpr float audioCutoff = 15000.0f;
    constexpr float pilotFastTrackingCutoff = 2500.0f;
    constexpr float pilotSlowTrackingCutoff = 150.0f;
    constexpr float stereoSeparationGain = 2.0f;
    constexpr float minSampleRate = 152000.0f;
    constexpr float maxSampleRate = 384000.0f;
}

StereoFmDecoder::StereoFmDecoder(unsigned int sampleRate):
    sampleRate(sampleRate),
    audioAlpha(alphaFor(sampleRate, audioCutoff)),
    pilotFastAlpha(alphaFor(sampleRate, pilotFastTrackingCutoff)),
    pilotSlowAlpha(alphaFor(sampleRate, pilotSlowTrackingCutoff)),
    phase19Step(2.0f * static_cast<float>(M_PI) * pilotFrequency / sampleRate),
    phase38Step(2.0f * static_cast<float>(M_PI) * stereoFrequency / sampleRate),
    startupSamplesRemaining(std::max<size_t>(sampleRate / 200, 1))
{
    if (sampleRate < minSampleRate) {
        throw std::runtime_error("sample rate too low for stereo FM decoding");
    }
    if (sampleRate > maxSampleRate) {
        throw std::runtime_error("sample rate too high for stereo FM decoding");
    }
}

float StereoFmDecoder::alphaFor(float sampleRate, float cutoff) {
    float alpha = 1.0f - std::exp(-2.0f * static_cast<float>(M_PI) * cutoff / sampleRate);
    return std::min(std::max(alpha, 0.0f), 1.0f);
}

bool StereoFmDecoder::canProcess() {
    std::lock_guard<std::mutex> lock(this->processMutex);
    return this->reader->available() > 0 && this->writer->writeable() >= 2;
}

void StereoFmDecoder::advancePhases() {
    phase19 += phase19Step;
    phase38 += phase38Step;
    while (phase19 > 2.0f * static_cast<float>(M_PI)) phase19 -= 2.0f * static_cast<float>(M_PI);
    while (phase38 > 2.0f * static_cast<float>(M_PI)) phase38 -= 2.0f * static_cast<float>(M_PI);
}

void StereoFmDecoder::process() {
    std::lock_guard<std::mutex> lock(this->processMutex);
    size_t samples = std::min(this->reader->available(), this->writer->writeable() / 2);
    float* input = this->reader->getReadPointer();
    float* output = this->writer->getWritePointer();

    for (size_t i = 0; i < samples; i++) {
        float x = input[i];
        if (std::isnan(x)) x = 0.0f;

        float sin19;
        float cos19;
        float sin38;
        float cos38;
        sincosf(phase19, &sin19, &cos19);
        sincosf(phase38, &sin38, &cos38);

        monoState += audioAlpha * (x - monoState);
        float compositeHigh = x - monoState;

        float pilotI = compositeHigh * cos19;
        float pilotQ = -compositeHigh * sin19;
        float pilotAlpha = startupSamplesRemaining > 0 ? pilotFastAlpha : pilotSlowAlpha;
        pilotIState += pilotAlpha * (pilotI - pilotIState);
        pilotQState += pilotAlpha * (pilotQ - pilotQState);

        float pilotNorm = pilotIState * pilotIState + pilotQState * pilotQState;
        if (pilotNorm > 1.0e-12f) {
            stereoPhaseCos = (pilotIState * pilotIState - pilotQState * pilotQState) / pilotNorm;
            stereoPhaseSin = 2.0f * pilotIState * pilotQState / pilotNorm;
        }

        float pilotEstimate = 2.0f * (pilotIState * cos19 - pilotQState * sin19);
        float stereoComposite = compositeHigh - pilotEstimate;
        float regeneratedCarrier = cos38 * stereoPhaseCos - sin38 * stereoPhaseSin;
        float diffMixed = 2.0f * stereoComposite * regeneratedCarrier;
        diffState += audioAlpha * (diffMixed - diffState);

        float left = 0.5f * (monoState + stereoSeparationGain * diffState);
        float right = 0.5f * (monoState - stereoSeparationGain * diffState);
        output[2 * i] = left;
        output[2 * i + 1] = right;

        advancePhases();
        if (startupSamplesRemaining > 0) startupSamplesRemaining--;
    }

    this->reader->advance(samples);
    this->writer->advance(samples * 2);
}

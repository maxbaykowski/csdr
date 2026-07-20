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

#pragma once

#include "module.hpp"

namespace Csdr {

    class StereoFmDecoder: public Module<float, float> {
        public:
            explicit StereoFmDecoder(unsigned int sampleRate);
            bool canProcess() override;
            void process() override;
        private:
            static float alphaFor(float sampleRate, float cutoff);
            void advancePhases();

            float sampleRate;
            float audioAlpha;
            float pilotFastAlpha;
            float pilotSlowAlpha;
            float phase19 = 0.0f;
            float phase38 = 0.0f;
            float phase19Step;
            float phase38Step;
            float monoState = 0.0f;
            float diffState = 0.0f;
            float pilotIState = 0.0f;
            float pilotQState = 0.0f;
            float stereoPhaseCos = 1.0f;
            float stereoPhaseSin = 0.0f;
            size_t startupSamplesRemaining;
    };

}

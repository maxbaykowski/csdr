/*
This software is part of libcsdr, a set of simple DSP routines for
Software Defined Radio.

Copyright (c) 2014, Andras Retzler <randras@sdr.hu>
Copyright (c) 2019-2021 Jakob Ketterl <jakob.ketterl@gmx.de>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ANDRAS RETZLER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "fractionaldecimator.hpp"

using namespace Csdr;

template <typename T>
FractionalDecimator<T>::FractionalDecimator(float rate, unsigned int num_poly_points, FirFilter<T, float> *filter, unsigned int channels):
    num_poly_points(num_poly_points &~ 1),
    poly_precalc_denomiator((float*) malloc(this->num_poly_points * sizeof(float))),
    xifirst(-(this->num_poly_points / 2) + 1),
    xilast(this->num_poly_points / 2),
    coeffs_buf((float*) malloc(this->num_poly_points * sizeof(float))),
    rate(rate),
    filter(filter),
    channels(channels == 0 ? 1 : channels)
{
    int id = 0; //index in poly_precalc_denomiator
    for (int xi = xifirst; xi <= xilast; xi++) {
        poly_precalc_denomiator[id] = 1;
        for(int xj = xifirst; xj <= xilast; xj++) {
            //poly_precalc_denomiator could be integer as well. But that would later add a necessary conversion.
            if (xi != xj) poly_precalc_denomiator[id] *= (xi - xj);
        }
        id++;
    }

    where = -xifirst;
}

template <typename T>
FractionalDecimator<T>::~FractionalDecimator() {
    free(poly_precalc_denomiator);
    free(coeffs_buf);
}

template <typename T>
bool FractionalDecimator<T>::canProcess() {
    std::lock_guard<std::mutex> lock(this->processMutex);
    size_t availableFrames = this->reader->available() / channels;
    size_t writeableFrames = this->writer->writeable() / channels;
    size_t size = std::min(
        availableFrames,
        static_cast<size_t>(ceil(static_cast<double>(writeableFrames) * rate))
    );
    size_t filterLen = filter != nullptr ? filter->getOverhead() : 0;
    return ceilf(where) + num_poly_points + filterLen < size;
}

template <typename T>
T FractionalDecimator<T>::filterSample(T* input, size_t frameIndex, unsigned int channel) {
    if (filter == nullptr) {
        return input[frameIndex * channels + channel];
    }

    if (channels == 1) {
        SparseView<T> sparse = filter->sparse(input);
        return sparse[frameIndex];
    }

    T acc = 0;
    auto* fir = static_cast<FirFilter<T, float>*>(filter);
    for (size_t tap = 0; tap <= fir->getOverhead(); tap++) {
        acc += input[(frameIndex + tap) * channels + channel] * fir->getTap(tap);
    }
    return acc;
}

template <typename T>
void FractionalDecimator<T>::process() {
    std::lock_guard<std::mutex> lock(this->processMutex);
    //This routine can handle floating point decimation rates.
    //It applies polynomial interpolation to samples that are taken into consideration from a pre-filtered input.
    //The pre-filter can be switched off by applying filter = nullptr.
    int oi = 0; //output index
    int index_high;
    size_t availableFrames = this->reader->available() / channels;
    size_t writeableFrames = this->writer->writeable() / channels;
    size_t size = std::min(
        availableFrames,
        static_cast<size_t>(ceil(static_cast<double>(writeableFrames) * rate))
    );
    size_t filterLen = filter != nullptr ? filter->getOverhead() : 0;
    T* input = this->reader->getReadPointer();
    T* output = this->writer->getWritePointer();
    //we optimize to calculate ceilf(where) only once every iteration, so we do it here:
    while ((index_high = ceilf(where)) + num_poly_points + filterLen < size) {
        // num_poly_points above is theoretically more than we could have here, but this makes the spectrum look good
        int index = index_high - 1;
        int id = 0;
        float xwhere = where - index;
        for (int xi = xifirst; xi <= xilast; xi++) {
            coeffs_buf[id] = 1;
            for (int xj = xifirst; xj <= xilast; xj++) {
                if (xi != xj) coeffs_buf[id] *= (xwhere - xj);
            }
            id++;
        }
        for (unsigned int channel = 0; channel < channels; channel++) {
            T acc = 0;
            for (int i = 0; i < num_poly_points; i++) {
                acc += (coeffs_buf[i] / poly_precalc_denomiator[i]) * filterSample(input, index + i, channel);
            }
            output[oi * channels + channel] = acc;
        }
        oi++;
        where += rate;
    }

    size_t input_processed = std::min(
        size,
        static_cast<size_t>(std::max(0, static_cast<int>(ceilf(where)) - 1))
    );
    where -= input_processed;

    this->reader->advance(input_processed * channels);
    this->writer->advance(oi * channels);
}

namespace Csdr {
    template class FractionalDecimator<float>;
    template class FractionalDecimator<complex<float>>;
}

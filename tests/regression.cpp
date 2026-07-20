#include "deemphasis.hpp"
#include "dcblock.hpp"
#include "filter.hpp"
#include "fir.hpp"
#include "fractionaldecimator.hpp"
#include "reader.hpp"
#include "stereofm.hpp"
#include "window.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

using namespace Csdr;

template <typename T>
class TestWriter: public Writer<T> {
    public:
        explicit TestWriter(size_t capacity):
            buffer(capacity),
            capacity(capacity)
        {}

        size_t writeable() override {
            return capacity;
        }

        T* getWritePointer() override {
            return buffer.data();
        }

        void advance(size_t how_much) override {
            collected.insert(collected.end(), buffer.begin(), buffer.begin() + how_much);
            advanced += how_much;
        }

        size_t advanced = 0;
        std::vector<T> collected;

    private:
        std::vector<T> buffer;
        size_t capacity;
};

class TestableWfmDeemphasis: public WfmDeemphasis {
    public:
        TestableWfmDeemphasis(unsigned int sampleRate, float tau): WfmDeemphasis(sampleRate, tau) {}
        using WfmDeemphasis::process;
};

static bool test_lowpass_single_output_progress() {
    auto window = std::unique_ptr<Window>(new HammingWindow());
    auto module = std::unique_ptr<FilterModule<float>>(new FilterModule<float>(new LowPassFilter<float>(0.01f, 0.05f, window.get())));

    std::vector<float> input(81, 0.0f);
    MemoryReader<float> reader(input.data(), input.size());
    TestWriter<float> writer(81);

    module->setReader(&reader);
    module->setWriter(&writer);

    if (!module->canProcess()) {
        std::cerr << "lowpass regression: expected filter to process with exactly one output sample worth of data\n";
        return false;
    }

    module->process();

    if (writer.advanced == 0) {
        std::cerr << "lowpass regression: expected at least one output sample at the FIR boundary\n";
        return false;
    }

    return true;
}

static bool test_fractionaldecimator_prefilter_high_rate_progress() {
    const float rate = 217.68707275390625f;
    const float transition = 0.03f;

    auto window = std::unique_ptr<Window>(new HammingWindow());
    auto filter = new LowPassFilter<complex<float>>(0.5f / (rate - transition), transition, window.get());
    auto module = std::unique_ptr<FractionalDecimator<complex<float>>>(new FractionalDecimator<complex<float>>(rate, 12, filter));

    std::vector<complex<float>> input(200);
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = complex<float>(std::sin(i * 0.01f), std::cos(i * 0.01f));
    }

    MemoryReader<complex<float>> reader(input.data(), input.size());
    TestWriter<complex<float>> writer(10240);

    module->setReader(&reader);
    module->setWriter(&writer);

    if (!module->canProcess()) {
        std::cerr << "fractionaldecimator regression: expected progress for high-rate prefiltered decimation\n";
        return false;
    }

    module->process();

    if (writer.advanced == 0) {
        std::cerr << "fractionaldecimator regression: expected at least one output sample\n";
        return false;
    }

    if (reader.available() > input.size()) {
        std::cerr << "fractionaldecimator regression: invalid reader state after processing\n";
        return false;
    }

    return true;
}

static bool test_nfm_deemphasis_8000_stays_bounded() {
    auto module = std::unique_ptr<NfmDeephasis>(new NfmDeephasis(8000));
    std::vector<float> input(256, 1.0f);
    MemoryReader<float> reader(input.data(), input.size());
    TestWriter<float> writer(256);

    module->setReader(&reader);
    module->setWriter(&writer);

    while (module->canProcess()) {
        module->process();
    }

    if (writer.collected.empty()) {
        std::cerr << "deemphasis regression: expected 8000 Hz NFM deemphasis to produce output\n";
        return false;
    }

    float max_abs = 0.0f;
    for (float sample : writer.collected) {
        if (!std::isfinite(sample)) {
            std::cerr << "deemphasis regression: 8000 Hz NFM deemphasis produced non-finite output\n";
            return false;
        }
        max_abs = std::max(max_abs, std::abs(sample));
    }

    if (max_abs > 2.0f) {
        std::cerr << "deemphasis regression: 8000 Hz NFM deemphasis output is implausibly large\n";
        return false;
    }

    return true;
}

static bool test_wfm_deemphasis_low_tau_stays_stable() {
    std::vector<float> input(4096, 1.0f);
    std::vector<float> output(input.size(), 0.0f);

    TestableWfmDeemphasis deemphasis40us(48000, 40e-6f);
    deemphasis40us.process(input.data(), output.data(), input.size());
    if (!std::isfinite(output.back()) || output.back() < 0.9f || output.back() > 1.0f) {
        std::cerr << "deemphasis regression: 40 us WFM deemphasis is unstable or has the wrong DC gain\n";
        return false;
    }

    TestableWfmDeemphasis deemphasis32us(48000, 32e-6f);
    std::fill(output.begin(), output.end(), 0.0f);
    deemphasis32us.process(input.data(), output.data(), input.size());
    if (!std::isfinite(output.back()) || output.back() < 0.9f || output.back() > 1.0f) {
        std::cerr << "deemphasis regression: 32 us WFM deemphasis is unstable or has the wrong DC gain\n";
        return false;
    }

    return true;
}

static double filteredToneEnergy(BandPassFilter<complex<float>>& filter, float tone, size_t outputSize) {
    size_t inputSize = outputSize + filter.getOverhead();
    std::vector<complex<float>> input(inputSize);
    std::vector<complex<float>> output(outputSize);

    for (size_t i = 0; i < input.size(); i++) {
        float phase = 2.0f * static_cast<float>(M_PI) * tone * i;
        input[i] = complex<float>(std::cos(phase), std::sin(phase));
    }

    filter.apply(input.data(), output.data(), outputSize);

    double energy = 0.0;
    for (const auto& sample : output) {
        energy += std::norm(static_cast<std::complex<float>>(sample));
    }

    return energy / output.size();
}

static double filteredToneEnergy(FftBandPassFilter& filter, float tone, size_t outputSize) {
    size_t inputSize = filter.getMinProcessingSize();
    std::vector<complex<float>> input(inputSize + filter.getOverhead());
    std::vector<complex<float>> output(outputSize);

    for (size_t i = 0; i < input.size(); i++) {
        float phase = 2.0f * static_cast<float>(M_PI) * tone * i;
        input[i] = complex<float>(std::cos(phase), std::sin(phase));
    }

    size_t produced = filter.apply(input.data(), output.data(), outputSize);

    double energy = 0.0;
    for (size_t i = 0; i < produced; i++) {
        energy += std::norm(static_cast<std::complex<float>>(output[i]));
    }

    return produced > 0 ? energy / produced : 0.0;
}

static double filteredFftToneEnergy(float lowcut, float highcut, Window& window, float tone) {
    FftBandPassFilter filter(lowcut, highcut, 0.05f, &window);
    return filteredToneEnergy(filter, tone, filter.getMinProcessingSize());
}

static bool test_bandpass_sideband_selection() {
    HammingWindow window;

    BandPassFilter<complex<float>> usbFilter(0.0f, 0.1f, 0.05f, &window);
    double usbPass = filteredToneEnergy(usbFilter, 0.05f, 2048);
    double usbReject = filteredToneEnergy(usbFilter, -0.05f, 2048);
    if (usbPass <= usbReject * 10.0) {
        std::cerr << "bandpass regression: USB filter does not prefer positive frequencies\n";
        return false;
    }

    BandPassFilter<complex<float>> lsbFilter(-0.1f, 0.0f, 0.05f, &window);
    double lsbPass = filteredToneEnergy(lsbFilter, -0.05f, 2048);
    double lsbReject = filteredToneEnergy(lsbFilter, 0.05f, 2048);
    if (lsbPass <= lsbReject * 10.0) {
        std::cerr << "bandpass regression: LSB filter does not prefer negative frequencies\n";
        return false;
    }

    double fftUsbPass = filteredFftToneEnergy(0.0f, 0.1f, window, 0.05f);
    double fftUsbReject = filteredFftToneEnergy(0.0f, 0.1f, window, -0.05f);
    if (fftUsbPass <= fftUsbReject * 10.0) {
        std::cerr << "bandpass regression: FFT USB filter does not prefer positive frequencies\n";
        return false;
    }

    double fftLsbPass = filteredFftToneEnergy(-0.1f, 0.0f, window, -0.05f);
    double fftLsbReject = filteredFftToneEnergy(-0.1f, 0.0f, window, 0.05f);
    if (fftLsbPass <= fftLsbReject * 10.0) {
        std::cerr << "bandpass regression: FFT LSB filter does not prefer negative frequencies\n";
        return false;
    }

    return true;
}

static bool test_dcblock_audio_removes_dc() {
    DcBlock dcblock(48000.0f, 15.0f, 0.05f);
    std::vector<float> input(48000, 1.0f);
    std::vector<float> output(input.size(), 0.0f);
    dcblock.process(input.data(), output.data(), input.size());

    if (!std::isfinite(output.back()) || std::abs(output.back()) > 0.01f) {
        std::cerr << "dcblock regression: audio-rate DC did not decay as expected\n";
        return false;
    }

    return true;
}

static bool test_dcblock_high_rate_preserves_near_center_tone() {
    const float sampleRate = 2400000.0f;
    const float toneHz = 1000.0f;
    DcBlock dcblock(sampleRate, 15.0f, 0.05f);

    std::vector<float> input(240000, 0.0f);
    std::vector<float> output(input.size(), 0.0f);
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = 0.25f + std::sin(2.0f * static_cast<float>(M_PI) * toneHz * i / sampleRate);
    }

    dcblock.process(input.data(), output.data(), input.size());

    double energy = 0.0;
    size_t start = input.size() / 2;
    for (size_t i = start; i < output.size(); i++) {
        if (!std::isfinite(output[i])) {
            std::cerr << "dcblock regression: high-rate output contains non-finite samples\n";
            return false;
        }
        energy += output[i] * output[i];
    }

    double rms = std::sqrt(energy / (output.size() - start));
    if (rms < 0.5) {
        std::cerr << "dcblock regression: high-rate near-center tone was attenuated too heavily\n";
        return false;
    }

    return true;
}

static bool test_stereofm_decoder_recovers_channels() {
    const unsigned int sampleRate = 240000;
    const size_t sampleCount = 240000 / 10;
    const float leftFreq = 1000.0f;
    const float rightFreq = 2300.0f;

    StereoFmDecoder decoder(sampleRate);
    std::vector<float> input(sampleCount);
    std::vector<float> output(sampleCount * 2, 0.0f);

    for (size_t i = 0; i < sampleCount; i++) {
        float t = static_cast<float>(i) / sampleRate;
        float left = 0.5f * std::sin(2.0f * static_cast<float>(M_PI) * leftFreq * t);
        float right = 0.5f * std::sin(2.0f * static_cast<float>(M_PI) * rightFreq * t);
        float mono = left + right;
        float diff = left - right;
        float pilot = 0.1f * std::cos(2.0f * static_cast<float>(M_PI) * 19000.0f * t);
        float stereo = diff * std::cos(2.0f * static_cast<float>(M_PI) * 38000.0f * t);
        input[i] = mono + pilot + stereo;
    }

    MemoryReader<float> reader(input.data(), input.size());
    TestWriter<float> writer(output.size());
    decoder.setReader(&reader);
    decoder.setWriter(&writer);

    while (decoder.canProcess()) {
        decoder.process();
    }

    if (writer.collected.size() != output.size()) {
        std::cerr << "stereofm regression: expected interleaved stereo output\n";
        return false;
    }

    double leftCorr = 0.0;
    double rightCorr = 0.0;
    double leftLeak = 0.0;
    double rightLeak = 0.0;
    double leftEnergy = 0.0;
    double rightEnergy = 0.0;
    size_t start = sampleRate / 200;

    for (size_t i = start; i < sampleCount; i++) {
        float t = static_cast<float>(i) / sampleRate;
        float leftRef = 0.5f * std::sin(2.0f * static_cast<float>(M_PI) * leftFreq * t);
        float rightRef = 0.5f * std::sin(2.0f * static_cast<float>(M_PI) * rightFreq * t);
        float leftOut = writer.collected[2 * i];
        float rightOut = writer.collected[2 * i + 1];
        leftCorr += leftOut * leftRef;
        rightCorr += rightOut * rightRef;
        leftLeak += leftOut * rightRef;
        rightLeak += rightOut * leftRef;
        leftEnergy += leftRef * leftRef;
        rightEnergy += rightRef * rightRef;
    }

    if (leftCorr / leftEnergy < 0.7 || rightCorr / rightEnergy < 0.7) {
        std::cerr << "stereofm regression: decoded channels do not match reference audio\n";
        return false;
    }

    if (std::abs(leftLeak / rightEnergy) > 0.12 || std::abs(rightLeak / leftEnergy) > 0.12) {
        std::cerr << "stereofm regression: stereo separation is too weak\n";
        return false;
    }

    try {
        StereoFmDecoder tooLow(96000);
        (void) tooLow;
        std::cerr << "stereofm regression: low sample rate should be rejected\n";
        return false;
    } catch (const std::runtime_error&) {
    }

    try {
        StereoFmDecoder tooHigh(500000);
        (void) tooHigh;
        std::cerr << "stereofm regression: high sample rate should be rejected\n";
        return false;
    } catch (const std::runtime_error&) {
    }

    return true;
}

int main() {
    if (!test_lowpass_single_output_progress()) return EXIT_FAILURE;
    if (!test_fractionaldecimator_prefilter_high_rate_progress()) return EXIT_FAILURE;
    if (!test_nfm_deemphasis_8000_stays_bounded()) return EXIT_FAILURE;
    if (!test_wfm_deemphasis_low_tau_stays_stable()) return EXIT_FAILURE;
    if (!test_bandpass_sideband_selection()) return EXIT_FAILURE;
    if (!test_dcblock_audio_removes_dc()) return EXIT_FAILURE;
    if (!test_dcblock_high_rate_preserves_near_center_tone()) return EXIT_FAILURE;
    if (!test_stereofm_decoder_recovers_channels()) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}

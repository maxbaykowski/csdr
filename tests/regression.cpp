#include "deemphasis.hpp"
#include "filter.hpp"
#include "fir.hpp"
#include "fractionaldecimator.hpp"
#include "reader.hpp"
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

int main() {
    if (!test_lowpass_single_output_progress()) return EXIT_FAILURE;
    if (!test_fractionaldecimator_prefilter_high_rate_progress()) return EXIT_FAILURE;
    if (!test_nfm_deemphasis_8000_stays_bounded()) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}

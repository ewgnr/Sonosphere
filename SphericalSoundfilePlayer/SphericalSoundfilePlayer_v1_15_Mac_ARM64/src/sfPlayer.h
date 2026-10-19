#pragma once

#include "ofMain.h"
#include "sndfile.hh"
#include <atomic>

// Optional windowing function
class WindowFunction 
{
public:
    enum class Type 
    {
        HANN = 0,
        BARTLETT,
        TRIANGLE,
        HAMMING,
        BLACKMAN,
        BLACKMAN_HARRIS,
        NUTTALL,
        FLAT_TOP
    };

    static double apply(Type type, unsigned long length, unsigned long pos) 
    {
        if (length < 2) return 0.0;

        double x = static_cast<double>(pos) / (length - 1);

        if (type == Type::HANN) return 0.5 * (1.0 - cos(2.0 * PI * x));
        if (type == Type::BARTLETT) return 1.0 - std::abs(2.0 * (pos - 0.5 * (length - 1)) / (length - 1));
        if (type == Type::TRIANGLE) return (2.0 / (length - 1.0)) * (((length - 1.0) / 2.0) - std::abs(pos - ((length - 1.0) / 2.0)));
        if (type == Type::HAMMING) return 0.54 - 0.46 * cos(2.0 * PI * x);
        if (type == Type::BLACKMAN) return 0.42 - 0.5 * cos(2.0 * PI * x) + 0.08 * cos(4.0 * PI * x);
        if (type == Type::BLACKMAN_HARRIS) return 0.35875 - 0.48829 * cos(2 * PI * x) + 0.14128 * cos(4 * PI * x) - 0.01168 * cos(6 * PI * x);
        if (type == Type::NUTTALL) return 0.355768 - 0.487396 * cos(2 * PI * x) + 0.144232 * cos(4 * PI * x) - 0.012604 * cos(6 * PI * x);
        if (type == Type::FLAT_TOP) return 1 - 1.93 * cos(2 * PI * x) + 1.29 * cos(4 * PI * x) - 0.388 * cos(6 * PI * x) + 0.032 * cos(8 * PI * x);

        return 1.0;
    }
};

class sfPlayer 
{
public:
    sfPlayer() = default;

    bool load(const std::string& path) 
    {
        SF_INFO info = { 0 };
        SNDFILE* file = sf_open(path.c_str(), SFM_READ, &info);

        std::cout << "Opened file " << path << std::endl;

        if (!file) {
            std::cerr << "Failed to open: " << path << "\n";
            return false;
        }

        soundInfo = info;
        samples.resize(soundInfo.frames * soundInfo.channels);
        sf_count_t readFrames = sf_readf_float(file, samples.data(), soundInfo.frames);
        sf_close(file);

        if (readFrames <= 0) 
        {
            std::cerr << "Failed to read samples from: " << path << "\n";
            return false;
        }

        numSamples = readFrames * soundInfo.channels;
        return true;
    }

    void start(double speed = 1.0, double start = 0.0, double end = 1.0) {
        playbackSpeed = speed;
        startPos = std::clamp(start, 0.0, 1.0);
        endPos = std::clamp(end, startPos, 1.0);
        position = startPos * numSamples;
        isActive = true;
    }

    void stop() {
        isActive = false;
    }

    double play(WindowFunction::Type windowType = WindowFunction::Type::HANN) 
    {
        if (!isActive || samples.empty()) return 0.0;

        position += playbackSpeed;
        if (position >= endPos * numSamples) {
            isActive = false;
            return 0.0;
        }

        size_t i = static_cast<size_t>(position);
        float frac = static_cast<float>(position - i);

        // Cubic interpolation
        int i0 = std::clamp(static_cast<int>(i) - 1, 0, static_cast<int>(samples.size() - 1));
        int i1 = std::clamp(static_cast<int>(i), 0, static_cast<int>(samples.size() - 1));
        int i2 = std::clamp(static_cast<int>(i) + 1, 0, static_cast<int>(samples.size() - 1));
        int i3 = std::clamp(static_cast<int>(i) + 2, 0, static_cast<int>(samples.size() - 1));

        float s0 = samples[i0];
        float s1 = samples[i1];
        float s2 = samples[i2];
        float s3 = samples[i3];

        float a = -0.5f * s0 + 1.5f * s1 - 1.5f * s2 + 0.5f * s3;
        float b = s0 - 2.5f * s1 + 2.0f * s2 - 0.5f * s3;
        float c = -0.5f * s0 + 0.5f * s2;
        float d = s1;

        float interpSample = ((a * frac + b) * frac + c) * frac + d;

        unsigned long windowLen = static_cast<unsigned long>((endPos - startPos) * numSamples);
        unsigned long windowPos = static_cast<unsigned long>(position - (startPos * numSamples));
        double envelope = WindowFunction::apply(windowType, windowLen, windowPos);

        return interpSample * envelope;
    }

    bool isPlaying() const {
        return isActive;
    }

    int getSampleRate() const {
        return soundInfo.samplerate;
    }

    int getNumChannels() const {
        return soundInfo.channels;
    }

    std::size_t getNumSamples() const {
        return samples.size();
    }

private:
    std::vector<float> samples;
    SF_INFO soundInfo = {};
    double position = 0.0;
    double playbackSpeed = 1.0;
    double startPos = 0.0;
    double endPos = 1.0;
    bool isActive = false;
    size_t numSamples = 0;
};

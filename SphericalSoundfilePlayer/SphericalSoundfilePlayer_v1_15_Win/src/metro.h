#pragma once

class Metro
{
public:
    Metro(float sampleRate = 48000.0) : sampleRate(sampleRate), phase(0.0), frequency(1.0), phaseIncrement(0), triggered(false)
    {}

    void setFrequency(double freq)
    {
        frequency = freq;
        phaseIncrement = frequency / sampleRate;
    }

    bool tick()
    {
        triggered = false;
        phase += phaseIncrement;

        if (phase >= 1.0) {
            phase -= 1.0;
            triggered = true;
        }

        return triggered;
    }

    bool isTriggered() const { return triggered; }

private:
    float sampleRate;
    float frequency;
    float phase;
    float phaseIncrement;
    bool triggered;
};

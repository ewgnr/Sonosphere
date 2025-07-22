#pragma once

class metro
{
public:
	metro() : phase(0), trigger(false) {}

	inline double play(double pFrq)
	{
		if (phase >= 1.0)
		{
			phase -= 1.0;
			trigger = true;
		}
		else
		{
			trigger = false;
		}
		double phaseInc = (1.0 / (48000 / (pFrq)));
		double pulse = phase < phaseInc ? 1.0 : 0.0;
		phase += phaseInc;
		return trigger;
	}

private:
	double phase;
	bool trigger;
};

#pragma once

#include "ofMain.h"
#include "sndfile.hh"
#include <atomic>

class windowFunctions
{
public:
	windowFunctions() : windowing(0) {}

	inline double play(const size_t& pSelWindow, const unsigned long& pWindowLength, const unsigned long& pWindowPos)
	{
		enum class windowSelector { HANN = 0, BARTLETT = 1, TRIANGLE = 2 };
		enum windowSelector actualWindow = (enum windowSelector)pSelWindow;

		if (actualWindow == windowSelector::HANN)
		{
			windowing = 0.5 * (1.0 - cos((2.0 * PI * pWindowPos) / (pWindowLength - 1)));
		}

		if (actualWindow == windowSelector::BARTLETT)
		{
			windowing = 1 - std::abs(2.0 * (pWindowPos - 0.5 * (pWindowLength - 1.0)) / (pWindowLength - 1.0));
		}

		if (actualWindow == windowSelector::TRIANGLE)
		{
			windowing = (2.0 / (pWindowLength - 1.0)) * (((pWindowLength - 1.0) / 2.0) - fabs(pWindowPos - ((pWindowLength - 1.0) / 2.0)));
		}

		return windowing;
	}

	double windowing;
};

class sfPlayer
{
public:
	sfPlayer() : playBackPosition(0), samples(0), numRead(0), soundInfo(), soundFile(), output(0), readSoundfileBuffer(false), speed(0), startPos(0), endPos(0) {}

	void read(std::string pPath)
	{
		soundInfo.format = 0;

		soundFile = sf_open(pPath.c_str(), SFM_READ, &soundInfo);

		std::cout << "Opened file " << pPath << std::endl;
		std::cout << "Format : " << soundInfo.format << std::endl;
		std::cout << "Sample Rate : " << soundInfo.samplerate << std::endl;
		std::cout << "Frames : " << soundInfo.frames << std::endl;
		std::cout << "Channels : " << soundInfo.channels << std::endl;

		samples.assign(soundInfo.frames * soundInfo.channels, 0);

		numRead = sf_readf_float(soundFile, &samples[0], soundInfo.frames);

		std::cout << "Loaded file with " << samples.size() << " samples." << std::endl;

		sf_close(soundFile);
	}

	void start(const double& pSpeed, const double& pStartPos, const double& pEndPos)
	{
		speed = pSpeed;
		startPos = pStartPos;
		playBackPosition = (numRead * soundInfo.channels) * pStartPos;
		endPos = pEndPos;
		readSoundfileBuffer = true;
	}

	void stop()
	{
		readSoundfileBuffer = false;
	}

	inline double play(const size_t pSelWindow)
	{
		if (readSoundfileBuffer)
		{
			playBackPosition = playBackPosition + ((speed * soundInfo.channels) / (48000 / soundInfo.samplerate));
			double remainder = playBackPosition - (long)playBackPosition;

			if ((long)playBackPosition >= (numRead * soundInfo.channels) * endPos)
			{
				output = 0;
				readSoundfileBuffer = false;
			}

			output = (double)(1 - remainder) * samples[1 + (long)playBackPosition] + remainder * samples[2 + (long)playBackPosition];
		}

		double envelope = window.play(pSelWindow, (numRead * soundInfo.channels) * (endPos - startPos), playBackPosition - (numRead * soundInfo.channels) * startPos);
		return output * envelope;
	}

	bool isPlaying() const
	{
		if (readSoundfileBuffer)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	double playBackPosition;
	std::vector<float> samples;
	sf_count_t numRead;
	SF_INFO soundInfo;
	SNDFILE* soundFile;
	double output;
	bool readSoundfileBuffer;
	windowFunctions window;
	double speed, startPos, endPos;
};
#pragma once

#include "ofMain.h"
#include "uvSphere.h"
#include "sfPlayer.h"
#include "metro.h"
#include "dab_osc_receiver.h"
#include "dab_osc_sender.h"
#include <memory>
#include "fileIO.h"
#include "ofxDatGui.h"
#include "ambi2dInPhase.h"
#include "convert3dCoordinates.h"

constexpr unsigned int mAudioSampleRate = 48000;
constexpr unsigned int mAudioBufferSize = 1024;
constexpr unsigned int mAudioNumOutputChannels = 2;

class ofApp : public ofBaseApp, public dab::OscListener
{
	public:
		void setup();
		void selectAudioDevice();
		void startAudioBackend();
		void setupOsc() throw (dab::Exception);
		void notify(std::shared_ptr<dab::OscMessage> pMessage);
		void updateOsc();
		void updateOsc(std::shared_ptr<dab::OscMessage> pMessage);
		void updateOscSender() throw (dab::Exception);
		void detectSensorCollisions(const glm::vec3& pSensorPosition);
		void detectJointCollisions(const std::vector<glm::vec3>& pJointPositions);
		void drawAudioTrigger(std::vector<std::size_t> pCollisionIndices);
		void triggerAudio(std::vector<std::size_t> pCollisionIndices);
		void calculate2dDistances(const std::vector<glm::vec3>& pSphereVertRefPoints);
		void update();
		void draw();
		void audioOut(ofSoundBuffer& outBuffer);
		void writeFileSetup(const std::string& pSfNumber);
		void writeFileToBuffer(const double& pInput);
		void writeFileToDisk(const std::vector<double>& pSampleBuffer);
		void keyPressed(int key);
		void onSliderEvent(ofxDatGuiSliderEvent e);
		void onDropdownEvent(ofxDatGuiDropdownEvent e);
		void onToggleEvent(ofxDatGuiToggleEvent e);
		void onButtonEvent(ofxDatGuiButtonEvent e);
		void onTextInputEvent(ofxDatGuiTextInputEvent e);
		void saveGuiValuesAsFile(const std::string& pGuiSelectPresetNumber);
		void readGuiValuesFromFile(const std::string& pGuiSelectPresetNumber);
		void exit();

		ofxDatGui* mGui;

		// RETRIGGER_AT_INDEX = true / false works with RETRIGGER_AT_END or CONTINOUS_TRIGGER
		bool RETRIGGER_AT_INDEX = false;
		bool RETRIGGER_AT_END = false;
		bool CONTINOUS_TRIGGER = false;	
		bool METRONOME_TRIGGER = false;
		bool GRAIN_SHUFFLE = false;
		
		double METRO_FREQUENCY = 1;
		double METRO_RAND_FREQ = 0;
		float PITCH_RAND_MIN = 0.5;
		float PITCH_RAND_MAX = 1.0;
		float SAMPLE_POS_START = 0.0;
		float SAMPLE_POS_END = 0.9;

		float MIN_COL_DISTANCE = 50;

		std::size_t RADIUS = 100;
		std::size_t ROWS = 10;
		std::size_t COLS = 10;
		std::size_t NUMBER_OF_SOUNDFILES = ROWS * COLS + 2 - 10;

		bool IMU_SENSOR = false;

		std::vector<string> mGuiFaderNames =
		{
			"METRO_FREQUENCY",
			"METRO_RAND_FREQ",
			"PITCH_RAND_MIN",
			"PITCH_RAND_MAX",
			"SAMPLE_POS_START",
			"SAMPLE_POS_END",
			"MIN_COL_DISTANCE"
		};

		vector<string> windows = 
		{ 
			"HANN", 
			"BARTLETT", 
			"TRIANGLE" 
		};

		int mGuiSelectWindow;
		bool mGuiUnlockPresets, mGuiReadOrWritePreset;
		std::string mGuiSelectPresetNumber;

		fileIO mGuiFileIO;

		uvSphere mVisSphere;
		ofEasyCam mVisCam;

		bool drawVerticies = true, drawMesh = true, drawIndicies = false, drawExitingPos = true;

		std::vector<std::string> fileNames;

		std::vector<std::string> directoryIterator(std::string pPath)
		{
			cout << pPath << endl;
			auto dirIter = std::filesystem::directory_iterator(pPath);

			for (auto& entry : dirIter)
			{
				std::cout << "entry path: " << entry.path().string() << std::endl;
				fileNames.push_back(entry.path().filename().string());
			}

			std::sort(fileNames.begin(), fileNames.end());

			return fileNames;
		}

		std::string selInputDevice;

		vector<ofSoundDevice> mDevices;
		ofSoundStream mSoundStream;
		ofSoundStreamSettings mSettings;

		sfPlayer* mAudioPlayers = new sfPlayer[NUMBER_OF_SOUNDFILES];
		ambiEncode2DThirdOrder* mAudioAmbiEncode = new ambiEncode2DThirdOrder[8];
		ambiDecode2DThirdOrder* mAudioAmbiDecode = new ambiDecode2DThirdOrder[8];
		
		metro mMetro;

		double mAudioPosMean, mAudioPosInDegree;
		std::size_t mAudioDistanceIndex;
		double mAudioOutput;

		SndfileHandle soundFileBuffer;
		std::vector<double> writeFileSampleBuffer;

		bool mAudioTrigger = false;
		unsigned int mAudioSoundStreamOnOffToggle = 0;
;
		std::vector<sfPlayer> mAudioGrains;
		metro mAudioPlayerMetro;

		std::atomic<bool> mAudioThreadSync = { false };

		dab::OscSender* mOscSender;
		std::string mOscSendAddress;
		int mOscSendPort;

		dab::OscReceiver* mOscReceiver;
		unsigned int mMaxOscMessageQueueLength = 1000;
		std::deque< std::shared_ptr<dab::OscMessage> > mOscMessageQueue;

		std::shared_ptr<ofApp> mSelf;
		std::mutex mOscLock;

		glm::quat mArduinoQuat;
		float x, y, z;

		std::vector<glm::vec3> mJointPositions = std::vector<glm::vec3>(32, glm::vec3(0, 0, 0));

		std::size_t exitingIndex;
		std::vector<std::size_t> mCollisionIndices;
		std::vector<std::size_t> mThreadedCollisionIndices;
		std::vector<std::size_t> mCollisionIndicesBefore;

		std::vector<std::size_t> set_difference(std::vector<std::size_t> const& pVector_1, std::vector<std::size_t> const& pVector_2)
		{
			std::vector <size_t> diff;
			std::set_difference(std::begin(pVector_1), std::end(pVector_1), std::begin(pVector_2), std::end(pVector_2), std::back_inserter(diff));
			return diff;
		}

		std::size_t colIndexTestCount = -1;

		convert3dCoordinates mConv3dCoord;
		std::vector<double> mCollisionRefVertDist;
};

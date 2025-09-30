#pragma once

#include "ofMain.h"
#include "uvSphere2.h"
#include "sfPlayer.h"
#include "metro.h"
#include "dab_osc_receiver.h"
#include "dab_osc_sender.h"
#include "ofxDatGui.h"
#include "ambi2dInPhase.h"
#include "vertexMapping.h"
#include "threadedVertexLoader.h"

#include <atomic>
#include <memory>

constexpr unsigned int mAudioSampleRate = 48000;
constexpr unsigned int mAudioBufferSize = 1024;
constexpr unsigned int mAudioNumOutputChannels = 2;

constexpr bool TEST_TRIGGER = false;

constexpr std::size_t MAX_GRAINS = 150;
constexpr std::size_t MAX_COL_POINTS = 20;
constexpr std::size_t RADIUS = 100;

class ofApp : public ofBaseApp, public dab::OscListener
{
public:
    // Core openFrameworks methods
    void setup();
    void update();
    void draw();
    void exit();
    void keyPressed(int key);
    void audioOut(ofSoundBuffer& outBuffer);

    // Audio / collision
    void selectAudioDevice();
    void startAudioBackend();
    void triggerAudio(std::vector<std::size_t> pCollisionIndices);
    void detectJointCollisions(const std::vector<glm::vec3>& pJointPositions);
    void drawAudioTrigger(const std::vector<std::size_t>& collisionIndices);

    // OSC Communication 
    void setupOsc() throw(dab::Exception);
    void notify(std::shared_ptr<dab::OscMessage> pMessage);
    void updateOsc();
    void updateOsc(std::shared_ptr<dab::OscMessage> pMessage);
    void updateOscSender() throw(dab::Exception);

    // File I/O
    void rebuildSphere(int xRes, int yRes);
    void writeFileSetup(const std::string& pSfNumber);
    void writeFileToBuffer(const double& pInput);
    void writeFileToDisk(const std::vector<double>& pSampleBuffer);
    bool forceFileReload = false;

    std::vector<std::string> directoryIterator(const std::string& pPath)
    {
        std::vector<std::string> fileNames;
        for (const auto& entry : std::filesystem::directory_iterator(pPath))
        {
            if (entry.is_regular_file())
            {
                fileNames.push_back(entry.path().filename().string());
            }
        }
        std::sort(fileNames.begin(), fileNames.end());
        return fileNames;
    }

    std::vector<std::string> getFolderNames()
    {
        std::vector<std::string> folders;
        ofDirectory dir(ofToDataPath("", true));
        dir.listDir();
        dir.sort();

        for (auto& sub : dir.getFiles())
        {
            if (sub.isDirectory() && ofIsStringInString(sub.getFileName(), "audio")) continue;
            if (sub.isDirectory() && isDigits(sub.getFileName()))
            {
                folders.push_back(sub.getFileName());
            }
        }
        return folders;
    }

    bool isDigits(const std::string& str) 
    {
        return std::all_of(str.begin(), str.end(), ::isdigit);
    }

    void loadSoundFolderByIndex(int index, bool forceReload, int xRes, int yRes);
    std::vector<std::string> availableAudioFolders;
    int currentLoadedFolderIndex = -1;

    // GUI Events
    void onTextInputEvent(ofxDatGuiTextInputEvent e);
    void onSliderEvent(ofxDatGuiSliderEvent e);
    void onDropdownEvent(ofxDatGuiDropdownEvent e);
    void onToggleEvent(ofxDatGuiToggleEvent e);
    void onButtonEvent(ofxDatGuiButtonEvent e);

    void saveGuiValuesAsFile(const std::string& pGuiSelectPresetNumber);
    void readGuiValuesFromFile(const std::string& pGuiSelectPresetNumber);

private:
    // GUI / Interface
    ofxDatGui* inst_GUI = nullptr;
    ofxDatGui* mesh_GUI = nullptr;
    ofxDatGui* presetControl_GUI = nullptr;
    int inst_GUI_SelectWindow = 0;
    WindowFunction::Type inst_WindowType = WindowFunction::Type::HANN;

    std::vector<std::string> inst_GUI_FaderNames = 
    {
        "METRO_FREQUENCY", "METRO_JITTER", "PITCH_RAND_MIN", "PITCH_RAND_MAX",
        "SAMPLE_POS_START", "SAMPLE_POS_END", "RAND_POS_STRENGTH", 
        "MIN_COL_DISTANCE"
    };

    std::vector<std::string> windows = 
    {
        "HANN", "BARTLETT", "TRIANGLE", "HAMMING",
        "BLACKMAN", "BLACKMAN_HARRIS", "NUTTALL", "FLAT_TOP"
    };

    bool mGuiUnlockPresets = false;
    bool mGuiReadOrWritePreset = false;
    std::string mGuiSelectPresetNumber;

    // Audio Engine
    bool loadingFinishedPrinted = false;

    std::vector<VertexMapping> mVertexMappings;
    ThreadedVertexLoader vertexLoader;

    struct Grain {
        std::shared_ptr<sfPlayer> player;
        glm::vec3 position;
    };
    std::vector<Grain> mAudioGrains;

    Metro mAudioPlayerMetro;

    float PITCH_RAND_MIN = 0.0f;
    float PITCH_RAND_MAX = 0.0f;
    float SAMPLE_POS_START = 0.0f;
    float SAMPLE_POS_END = 0.0f;
    float RAND_POS_STRENGTH = 0.0f;
    float MIN_COL_DISTANCE = 1.0f;
    float METRO_FREQUENCY = 0.0f;
    float METRO_JITTER = 0.0f;

    std::atomic<bool> RETRIGGER_AT_INDEX{ true };
    std::atomic<bool> RETRIGGER_AT_END{ false };
    std::atomic<bool> METRONOME_TRIGGER{ false };
    std::atomic<bool> GRAIN_SHUFFLE{ true };

    std::atomic<bool> mAudioThreadSync = { false };
    bool mAudioRecording = false;
    bool mAudioTrigger = true;
    unsigned int mAudioSoundStreamOnOffToggle = 0;

    ofSoundStream mSoundStream;
    ofSoundStreamSettings mSettings;
    std::vector<ofSoundDevice> mDevices;
    std::string selInputDevice;

    std::pair<float, float> getRandomizedGrainBounds(float startNorm, float endNorm, float randStrength)
    {
        float grainRangeNorm = endNorm - startNorm;
        float fullRandomOffset = ofRandom(0.0f, grainRangeNorm);
        float blendedOffset = fullRandomOffset * randStrength;
        float grainStartNorm = std::min(startNorm + blendedOffset, endNorm);
        float grainEndNorm = endNorm;

        return { grainStartNorm, grainEndNorm };
    }

    // Ambisonics
    ambiEncode2DThirdOrder mAudioAmbiEncoder;
    std::vector<ambiDecode2DThirdOrder> mAudioAmbiDecoder;

    double mAudioPosMean = 0.0;
    double mAudioPosInDegree = 0.0;
    std::size_t mAudioDistanceIndex = 0;

    // OSC
    dab::OscSender* mOscSender = nullptr;
    dab::OscReceiver* mOscReceiver = nullptr;
    std::string mOscSendAddress;
    int mOscSendPort = 0;

    unsigned int mMaxOscMessageQueueLength = 1000;
    std::deque<std::shared_ptr<dab::OscMessage>> mOscMessageQueue;
    std::mutex mOscLock;

    // File / Sample
    SndfileHandle soundFileBuffer;
    std::vector<double> writeFileSampleBuffer;
    std::vector<std::string> fileNames;

    // Collision / Motion
    bool USE_IMU_SENSOR = false;
    bool USE_BOID_COLLISION = false;
    bool USE_MOCAP_DATA = false;

    bool RNN_MOTION_CONTINUATION = true;

    std::vector<glm::vec3> mJointPositions = std::vector<glm::vec3>(24, glm::vec3(0));
    std::vector<std::size_t> mCollisionIndices;
    std::vector<std::size_t> mThreadedCollisionIndices;
    std::vector<std::size_t> mCollisionIndicesBefore;
    std::vector<std::size_t> mManualCollisionIndices;
    std::vector<double> mCollisionRefVertDist;

    std::size_t colIndexTestCount = -1;
    std::size_t exitingIndex = 0;

    // 3D / Visualization
    std::unique_ptr<uvSphere> sphere;
    ofEasyCam cam;

    glm::quat mArduinoQuat;
    float x = 0.0f, y = 0.0f, z = 0.0f;

    bool drawVerticies = true;
    bool drawMesh = true;
    bool drawIndicies = false;
    bool drawExitingPos = true;
    bool meshBuilt = false;

    // Misc
    std::shared_ptr<ofApp> mSelf;
    int currentXRes = 10;
    int currentYRes = 10;

    // Boids
    std::vector<glm::vec3> mBoidPositions;
};
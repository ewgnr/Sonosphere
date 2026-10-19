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
#include <filesystem>
#include <algorithm>
#include <mutex>
#include <deque>

constexpr unsigned int mAudioSampleRate = 48000;
constexpr unsigned int mAudioBufferSize = 1024;

constexpr bool TEST_TRIGGER = false;

constexpr std::size_t MAX_GRAINS = 150;
constexpr std::size_t MAX_COL_POINTS = 20;
constexpr std::size_t RADIUS = 100;

class ofApp : public ofBaseApp, public dab::OscListener
{
public:
    // Core openFrameworks methods
    void setup() override;
    void update() override;
    void draw() override;
    void exit() override;
    void keyPressed(int key) override;
    void audioOut(ofSoundBuffer& outBuffer) override;

    // Audio / collision
    void selectAudioDevice();
    void startAudioBackend();
    void triggerAudio(std::vector<std::size_t> pCollisionIndices);
    void detectJointCollisions(const std::vector<glm::vec3>& pJointPositions);
    void clearAllAudio();

    // OSC Communication 
    void setupOsc() throw(dab::Exception);
    void notify(std::shared_ptr<dab::OscMessage> pMessage) override;
    void updateOsc();
    void handleOscMessage(std::shared_ptr<dab::OscMessage> pMessage);
    void handleMocapJointMessage(const std::vector<dab::_OscArg*>& arguments);
    void handleBoidPositions(const std::vector<dab::_OscArg*>& arguments);
    void handleImuMessage(const std::vector<dab::_OscArg*>& arguments);
    void handleAudioPresetMessage(const std::vector<dab::_OscArg*>& arguments);
    void updateOscSender() throw(dab::Exception);

    // File I/O
    void queryInitialOptions();
    void rebuildSphere(int xRes, int yRes);
    void updatePresetTrigger();
    void updateMeshLoader();
    void notifyMeshReady();
    void writeFileSetup(const std::string& pSfNumber);
    void writeFileToBuffer(const double& pInput);
    void writeFileToDisk(const std::vector<double>& pSampleBuffer);
    void loadSoundFolderByIndex(int index, bool forceReload = false, int xRes = 10, int yRes = 10);
    bool forceFileReload = false;

    // GUI Events
    void setupGui();
    void onTextInputEvent(ofxDatGuiTextInputEvent e);
    void onSliderEvent(ofxDatGuiSliderEvent e);
    void onDropdownEvent(ofxDatGuiDropdownEvent e);
    void onToggleEvent(ofxDatGuiToggleEvent e);
    void onButtonEvent(ofxDatGuiButtonEvent e);

    void saveGuiValuesAsFile(const std::string& pGuiSelectPresetNumber);
    void readGuiValuesFromFile(const std::string& pGuiSelectPresetNumber);

    // Utilities
    std::string apiToString(ofSoundDevice::Api api) const {
        switch (api) {
        case ofSoundDevice::Api::UNSPECIFIED: return "UNSPECIFIED";
        case ofSoundDevice::Api::DEFAULT:     return "DEFAULT";
        case ofSoundDevice::Api::ALSA:        return "ALSA";
        case ofSoundDevice::Api::PULSE:       return "PULSE";
        case ofSoundDevice::Api::OSS:         return "OSS";
        case ofSoundDevice::Api::JACK:        return "JACK";
        case ofSoundDevice::Api::OSX_CORE:    return "OSX_CORE";
        case ofSoundDevice::Api::MS_WASAPI:   return "MS_WASAPI";
        case ofSoundDevice::Api::MS_ASIO:     return "MS_ASIO";
        case ofSoundDevice::Api::MS_DS:       return "MS_DS";
        default:                              return "UNKNOWN_API";
        }
    }

    bool isDigits(const std::string& str) const {
        return std::all_of(str.begin(), str.end(), ::isdigit);
    }

    std::vector<std::string> directoryIterator(const std::string& path) const {
        std::vector<std::string> fileNames;
        for (const auto& entry : std::filesystem::directory_iterator(path))
        {
            if (entry.is_regular_file())
            {
                fileNames.push_back(entry.path().filename().string());
            }
        }
        std::sort(fileNames.begin(), fileNames.end());
        return fileNames;
    }

    std::vector<std::string> getFolderNames() const {
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

    std::pair<float, float> getRandomizedGrainBounds(float startNorm, float endNorm, float randStrength) {
        float grainRangeNorm = endNorm - startNorm;
        float fullRandomOffset = ofRandom(0.0f, grainRangeNorm);
        float blendedOffset = fullRandomOffset * randStrength;
        float grainStartNorm = std::min(startNorm + blendedOffset, endNorm);
        float grainEndNorm = endNorm;

        return { grainStartNorm, grainEndNorm };
    }

private:
    // GUI / Interface
    std::unique_ptr<ofxDatGui> inst_GUI;
    std::unique_ptr<ofxDatGui> mesh_GUI;
    std::unique_ptr<ofxDatGui> presetControl_GUI;
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

    bool pendingPresetLoad = false;
    std::string pendingPresetStr;
    bool pendingAudioTrigger = false;

    // Audio Engine
    void triggerAtIndexOrEnd(const std::vector<std::size_t>& indices);
    void triggerWithMetronome(const std::vector<std::size_t>& indices);

    bool loadingFinishedPrinted = false;

    std::vector<VertexMapping> mVertexMappings;
    std::unordered_map<std::size_t, std::size_t> mVertexMap;
    ThreadedVertexLoader vertexLoader;

    struct Grain 
    {
        std::shared_ptr<sfPlayer> player;
        glm::vec3 position;
    };
    std::vector<Grain> mAudioGrains;
    std::shared_ptr<ofApp::Grain> createGrainFromVertex(std::size_t index);

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
    unsigned int mAudioTriggerOnOffToggle = 0;

    ofSoundStream mSoundStream;
    ofSoundStreamSettings mSettings;
    std::vector<ofSoundDevice> mDevices;
    std::string selInputDevice;
    unsigned int mAudioNumOutputChannels = 2;

    // Ambisonics
    ambiEncode2DThirdOrder mAudioAmbiEncoder;
    std::vector<ambiDecode2DThirdOrder> mAudioAmbiDecoder;

    double mAudioPosMean = 0.0;
    double mAudioPosInDegree = 0.0;
    std::size_t mAudioDistanceIndex = 0;

    // OSC
    std::unique_ptr<dab::OscSender> mOscSender;
    std::unique_ptr<dab::OscReceiver> mOscReceiver;
    std::string mOscSendAddress;
    int mOscSendPort = 0;

    unsigned int mMaxOscMessageQueueLength = 1000;
    std::deque<std::shared_ptr<dab::OscMessage>> mOscMessageQueue;
    std::mutex mOscLock;

    // File / Sample
    SndfileHandle soundFileBuffer;
    std::vector<double> writeFileSampleBuffer;
    std::vector<std::string> fileNames;
    std::vector<std::string> availableAudioFolders;
    int currentLoadedFolderIndex = -1;

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
    void setupRenderOptions();

    void setupSphere();
    void drawSphereMesh();
    void drawAudioTrigger(const std::vector<std::size_t>& collisionIndices);
    void drawCollisionPoints();
    void drawImuSensor();
    void drawBoidPositions();
    void drawMocapJoints();
    std::unique_ptr<uvSphere> sphere;
    bool drawVerticies = true;
    bool drawMesh = true;
    bool drawIndicies = false;
    bool drawExitingPos = true;
    bool meshBuilt = false;

    void setupCamera();
    ofEasyCam cam;

    glm::quat mArduinoQuat;
    float x = 0.0f, y = 0.0f, z = 0.0f;

    // Misc
    void setupLogging();
    std::shared_ptr<ofApp> mSelf;
    int currentXRes = 10;
    int currentYRes = 10;

    // Boids
    std::vector<glm::vec3> mBoidPositions;
};

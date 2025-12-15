#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setupLogging()
{
    ofSetLogLevel(OF_LOG_VERBOSE);
    ofSetLoggerChannel(std::make_shared<ofConsoleLoggerChannel>());
}

//--------------------------------------------------------------
int askIntInRange(const std::string& prompt, int minVal, int maxVal, int defaultVal)
{
    std::string input;
    int value;

    while (true)
    {
        std::cout << prompt << " [" << minVal << "-" << maxVal << ", default: " << defaultVal << "]: ";
        std::getline(std::cin, input);

        if (input.empty())
            return defaultVal;

        try {
            value = std::stoi(input);
            if (value < minVal || value > maxVal) {
                std::cerr << "Value out of range. Try again.\n";
            }
            else {
                return value;
            }
        }
        catch (const std::exception&) {
            std::cerr << "Invalid input. Please enter a number.\n";
        }
    }
}

//--------------------------------------------------------------
void ofApp::queryInitialOptions()
{
    std::cout << "Select ONE of the following input modes:\n";
    std::cout << "[0] IMU Sensor\n";
    std::cout << "[1] Boid Collision (Flocking)\n";
    std::cout << "[2] Motion Capture (MoCap) Data\n";

    int choice = askIntInRange("Enter choice", 0, 2, 1);

    USE_IMU_SENSOR = (choice == 0);
    USE_BOID_COLLISION = (choice == 1);
    USE_MOCAP_DATA = (choice == 2);

    std::cout << "Mode selected: "
        << (USE_IMU_SENSOR ? "IMU Sensor" :
            USE_BOID_COLLISION ? "Boid Collision" :
            "MoCap Data")
        << "\n";
}

//--------------------------------------------------------------
void ofApp::selectAudioDevice()
{
    std::vector<ofSoundDevice::Api> apis;
    for (int i = static_cast<int>(ofSoundDevice::Api::DEFAULT);
        i < static_cast<int>(ofSoundDevice::Api::NUM_APIS); ++i)
    {
        apis.push_back(static_cast<ofSoundDevice::Api>(i));
    }

    std::cout << "Available Audio APIs:\n";
    for (size_t i = 0; i < apis.size(); ++i)
    {
        std::cout << "[" << i << "] " << apiToString(apis[i]) << "\n";
    }

    int selectedApiIndex = askIntInRange("Select audio API by index", 0, apis.size() - 1, 0);
    ofSoundDevice::Api selectedApi = apis[selectedApiIndex];
    mSettings.setApi(selectedApi);

    mDevices = mSoundStream.getDeviceList(selectedApi);

    std::cout << "\nAvailable devices for API " << ofToString(selectedApi) << ":\n";
    for (int i = 0; i < mDevices.size(); ++i)
    {
        std::cout << "[" << i << "] " << mDevices[i].name << "\n";
    }

    int selInputDeviceInt = askIntInRange("Select input device by index", 0, mDevices.size() - 1, 0);
    selInputDevice = std::to_string(selInputDeviceInt);

    mAudioNumOutputChannels = askIntInRange("Select number of audio output channels", 2, 8, 2);
    std::cout << "Selected output channels: " << mAudioNumOutputChannels << "\n";
}

//--------------------------------------------------------------
void ofApp::startAudioBackend()
{
    mSettings.setOutDevice(mDevices[std::stoi(selInputDevice)]);
    mSettings.setInDevice(mDevices[std::stoi(selInputDevice)]);
    mSettings.numOutputChannels = mAudioNumOutputChannels;
    mSettings.numInputChannels = 0;
    mSettings.sampleRate = mAudioSampleRate;
    mSettings.bufferSize = mAudioBufferSize;
    mSettings.numBuffers = 4;
    mSettings.setOutListener(this);
    mSettings.setInListener(this);
    mSoundStream.setup(mSettings);
}

//--------------------------------------------------------------
void ofApp::setupGui()
{
    inst_GUI = std::make_unique<ofxDatGui>(ofxDatGuiAnchor::TOP_LEFT);
    inst_GUI->addSlider(inst_GUI_FaderNames[0], 0.001, 20.0)->setPrecision(7);
    inst_GUI->addSlider(inst_GUI_FaderNames[1], 0.0, 1.0)->setPrecision(7);
    inst_GUI->addSlider(inst_GUI_FaderNames[2], 0.001, 12.0)->setPrecision(7);
    inst_GUI->addSlider(inst_GUI_FaderNames[3], 0.001, 12.0)->setPrecision(7);
    inst_GUI->addSlider(inst_GUI_FaderNames[4], 0.0, 1.0)->setPrecision(7);
    inst_GUI->addSlider(inst_GUI_FaderNames[5], 0.0, 1.0)->setPrecision(7);
    inst_GUI->addSlider(inst_GUI_FaderNames[6], 0.0, 1.0)->setPrecision(7);
    inst_GUI->addSlider(inst_GUI_FaderNames[7], 1.0, 100.0)->setPrecision(7);

    inst_GUI->addToggle("RETRIGGER_AT_INDEX", RETRIGGER_AT_INDEX);
    inst_GUI->addToggle("RETRIGGER_AT_END", RETRIGGER_AT_END);
    inst_GUI->addToggle("METRONOME_TRIGGER", METRONOME_TRIGGER);
    inst_GUI->addToggle("GRAIN_SHUFFLE", GRAIN_SHUFFLE);
    inst_GUI->addDropdown("SELECT_WINDOW_FUNCTION", windows);
    inst_GUI->addTextInput("SELECT_FOLDER", "0")->setInputType(ofxDatGuiInputType::NUMERIC);
    inst_GUI->addTextInput("X_RES", "10")->setInputType(ofxDatGuiInputType::NUMERIC);
    inst_GUI->addTextInput("Y_RES", "10")->setInputType(ofxDatGuiInputType::NUMERIC);

    inst_GUI->onSliderEvent(this, &ofApp::onSliderEvent);
    inst_GUI->onDropdownEvent(this, &ofApp::onDropdownEvent);
    inst_GUI->onToggleEvent(this, &ofApp::onToggleEvent);
    inst_GUI->onButtonEvent(this, &ofApp::onButtonEvent);
    inst_GUI->onTextInputEvent(this, &ofApp::onTextInputEvent);

    presetControl_GUI = std::make_unique<ofxDatGui>(ofxDatGuiAnchor::BOTTOM_LEFT);
    presetControl_GUI->addToggle("UNLOCK_PRESET");
    presetControl_GUI->addButton("READ_WRITE_PRST");
    presetControl_GUI->addTextInput("SEL_PRST_NR")->setInputType(ofxDatGuiInputType::NUMERIC);

    presetControl_GUI->onToggleEvent(this, &ofApp::onToggleEvent);
    presetControl_GUI->onButtonEvent(this, &ofApp::onButtonEvent);
    presetControl_GUI->onTextInputEvent(this, &ofApp::onTextInputEvent);

    mesh_GUI = std::make_unique<ofxDatGui>(ofxDatGuiAnchor::TOP_RIGHT);
    mesh_GUI->enableFboMode(true, ofGetWidth(), ofGetHeight());
    mesh_GUI->setWidth(180, 0.6f);
    mesh_GUI->onTextInputEvent(this, &ofApp::onTextInputEvent);
}

//--------------------------------------------------------------
void ofApp::setupSphere()
{
    sphere = std::make_unique<uvSphere>(10, 10, 1.0f);

    const auto& vertices = sphere->getUniqueVertices();
    mVertexMappings.clear();
    mVertexMap.clear();

    for (std::size_t vertexIdx = 0; vertexIdx < vertices.size(); ++vertexIdx)
    {
        VertexMapping vm;
        vm.guiIndex = vertexIdx;
        vm.vertexIndex = vertexIdx;
        vm.position = vertices[vertexIdx];

        auto input = mesh_GUI->addTextInput("V " + ofToString(vertexIdx), "100.0");
        input->setInputType(ofxDatGuiInputType::NUMERIC);
        vm.guiInput = input;

        mVertexMappings.push_back(vm);
        mVertexMap[vm.vertexIndex] = mVertexMappings.size() - 1;
    }
}

//--------------------------------------------------------------
void ofApp::setupCamera()
{
    cam.setDistance(500);
    cam.removeAllInteractions();
    cam.addInteraction(ofEasyCam::TRANSFORM_ROTATE, OF_MOUSE_BUTTON_RIGHT);
    cam.setLensOffset(ofVec2f(-0.3, 0));
}

//--------------------------------------------------------------
void ofApp::setupOsc()
{
    try
    {
        mSelf = std::shared_ptr<ofApp>(this);
        mOscReceiver = std::make_unique<dab::OscReceiver>("MocapReceiver", 9004);
        mOscReceiver->registerOscListener(std::weak_ptr<ofApp>(mSelf));
        mOscReceiver->start();

        mOscSendAddress = "127.0.0.1";
        mOscSendPort = 9010;
        mOscSender = std::make_unique<dab::OscSender>("MocapSender", mOscSendAddress, mOscSendPort);
    }
    catch (dab::Exception& e)
    {
        std::cout << e << "\n";
    }
}

//--------------------------------------------------------------
void ofApp::setupRenderOptions()
{
    ofSetVerticalSync(true);
    ofBackground(0);
}

//--------------------------------------------------------------
void ofApp::setup()
{
    setupLogging();
    queryInitialOptions();
    selectAudioDevice();

    mAudioGrains.reserve(MAX_GRAINS);
    mAudioAmbiDecoder.resize(mAudioNumOutputChannels);

    availableAudioFolders = getFolderNames();

    if (availableAudioFolders.empty()) {
        ofLogWarning() << "[setup] No audio folders found inside data/";
        return;
    }

    setupGui();
    setupSphere();
    setupCamera();
    setupOsc();

    loadSoundFolderByIndex(0, forceFileReload, currentXRes, currentYRes);

    setupRenderOptions();
}

//--------------------------------------------------------------
void ofApp::rebuildSphere(int xRes, int yRes)
{
    vertexLoader.stop(); // stop async loading safely before deleting GUI

    sphere = std::make_unique<uvSphere>(xRes, yRes, 1.0f);

    mesh_GUI.reset();
    mesh_GUI = std::make_unique<ofxDatGui>(ofxDatGuiAnchor::TOP_RIGHT);
    mesh_GUI->enableFboMode(true, ofGetWidth(), ofGetHeight());
    mesh_GUI->setWidth(180, 0.6f);
    mesh_GUI->onTextInputEvent(this, &ofApp::onTextInputEvent);

    mVertexMappings.clear();
    mVertexMap.clear();
    const auto& vertices = sphere->getUniqueVertices();

    for (std::size_t vertexIdx = 0; vertexIdx < vertices.size(); ++vertexIdx) {
        VertexMapping vm;
        vm.guiIndex = vertexIdx;
        vm.vertexIndex = vertexIdx;
        vm.position = vertices[vertexIdx];

        auto input = mesh_GUI->addTextInput("V " + ofToString(vertexIdx), "100.0");
        input->setInputType(ofxDatGuiInputType::NUMERIC);
        vm.guiInput = input;


        mVertexMappings.push_back(vm);
        mVertexMap[vm.vertexIndex] = mVertexMappings.size() - 1;
    }

    ofLogNotice() << "[rebuildSphere] Rebuilt with " << vertices.size() << " vertices.";
}

//--------------------------------------------------------------
void ofApp::notify(std::shared_ptr<dab::OscMessage> pMessage)
{
    mOscLock.lock();

    mOscMessageQueue.push_back(pMessage);
    if (mOscMessageQueue.size() > mMaxOscMessageQueueLength) mOscMessageQueue.pop_front();

    mOscLock.unlock();
}

//--------------------------------------------------------------
void ofApp::handleMocapJointMessage(const std::vector<dab::_OscArg*>& arguments)
{
    size_t jointCount = mJointPositions.size();
    size_t argCount = arguments.size();
    size_t rootJointIndex = 0;

    glm::vec3 rootJointPosition;

    if (RNN_MOTION_CONTINUATION)
    {
        rootJointPosition.x = *arguments[rootJointIndex * 3];
        rootJointPosition.y = *arguments[rootJointIndex * 3 + 2];
        rootJointPosition.z = *arguments[rootJointIndex * 3 + 1];

        rootJointPosition.x *= -1.0;
    }
    else
    { 
        rootJointPosition.x = *arguments[rootJointIndex * 3];
        rootJointPosition.y = *arguments[rootJointIndex * 3 + 1];
        rootJointPosition.z = *arguments[rootJointIndex * 3 + 2];
    }

    for (int jI = 0, aI = 0; jI < jointCount; ++jI, aI += 3)
    {
        if (RNN_MOTION_CONTINUATION)
        {
            mJointPositions[jI].x = *arguments[aI];
            mJointPositions[jI].y = *arguments[aI + 2];
            mJointPositions[jI].z = *arguments[aI + 1];

            mJointPositions[jI].x *= -1.0;
        }
        else
        {
            mJointPositions[jI].x = *arguments[aI];
            mJointPositions[jI].y = *arguments[aI + 1];
            mJointPositions[jI].z = *arguments[aI + 2];
        }

        mJointPositions[jI].x -= rootJointPosition.x;
        mJointPositions[jI].y -= rootJointPosition.y;
        mJointPositions[jI].z -= rootJointPosition.z;
    }
}

//--------------------------------------------------------------
void ofApp::handleBoidPositions(const std::vector<dab::_OscArg*>& arguments)
{
    size_t argCount = arguments.size();
    size_t numBoids = argCount / 3;

    std::vector<glm::vec3> tempBoidPositions(numBoids);

    for (size_t i = 0, bI = 0; i < numBoids; ++i, bI += 3)
    {
        glm::vec3 pos;
        pos.x = *arguments[bI];
        pos.y = *arguments[bI + 1];
        pos.z = *arguments[bI + 2];

        tempBoidPositions[i] = pos;
    }

    float maxLength = 0.0001f;
    for (const auto& pos : tempBoidPositions)
    {
        float len = glm::length(pos);
        if (len > maxLength)
            maxLength = len;
    }

    float targetRadius = 100.0f;
    for (auto& pos : tempBoidPositions)
    {
        pos = glm::normalize(pos) * targetRadius;
    }

    mBoidPositions = std::move(tempBoidPositions);
}

//--------------------------------------------------------------
void ofApp::handleImuMessage(const std::vector<dab::_OscArg*>& arguments)
{
    float w = *arguments[0];
    float x = *arguments[1];
    float y = *arguments[2];
    float z = *arguments[3];
    mArduinoQuat = glm::quat(w, x, y, z);
}

void ofApp::handleAudioPresetMessage(const std::vector<dab::_OscArg*>& arguments)
{
    int presetNumber = *arguments[0];
    pendingPresetStr = std::to_string(presetNumber);
    pendingPresetLoad = true;

    int presetAudioTrigger = *arguments[1];
    pendingAudioTrigger = bool(presetAudioTrigger);
}

//--------------------------------------------------------------
void ofApp::updateOsc()
{
    std::scoped_lock lock(mOscLock);

    while (!mOscMessageQueue.empty())
    {
        auto oscMessage = mOscMessageQueue.front();
        handleOscMessage(oscMessage);
        mOscMessageQueue.pop_front();
    }
}

//--------------------------------------------------------------
void ofApp::handleOscMessage(std::shared_ptr<dab::OscMessage> pMessage)
{
    try {
        const std::string& address = pMessage->address();
        const std::vector<dab::_OscArg*>& arguments = pMessage->arguments();

        if (address == "/mocap/joint/pos_world" || address == "/mocap/0/joint/pos_world")
        {
            handleMocapJointMessage(arguments);
        }
        else if (address == "/boid/positions")
        {
            handleBoidPositions(arguments);
        }
        else if (address == "/imu/2")
        {
            handleImuMessage(arguments);
        }
        else if (address == "/audio/preset")
        {
            handleAudioPresetMessage(arguments);
        }
    }
    catch (dab::Exception& e) {
        std::cout << e << "\n";
    }
}

//--------------------------------------------------------------
void ofApp::updateOscSender() throw (dab::Exception)
{
    try
    {
        std::string messageAddress = "/mocap/joint/pos_world";
        std::shared_ptr<dab::OscMessage> message(new dab::OscMessage(messageAddress));

        for (int pI = 0; pI < mJointPositions.size(); ++pI)
        {
            message->add(mJointPositions[pI].x);
            message->add(mJointPositions[pI].y);
            message->add(mJointPositions[pI].z);
        }
            mOscSender->send(message);            
    }
    catch (dab::Exception& e)
    {
        std::cout << e << "\n";
    }
}

//--------------------------------------------------------------
void ofApp::updatePresetTrigger()
{
    if (!pendingPresetLoad) return;

    bool triggerChanged = (mAudioTrigger != pendingAudioTrigger);
    mAudioTrigger = pendingAudioTrigger;
    std::cout << "Trigger: " << mAudioTrigger << std::endl;

    if (triggerChanged && mAudioTrigger)
    {
        clearAllAudio();
    }

    presetControl_GUI->getTextInput("SEL_PRST_NR")->setText(pendingPresetStr);
    mGuiSelectPresetNumber = pendingPresetStr;

    if (auto btn = presetControl_GUI->getButton("READ_WRITE_PRST")) {
        onButtonEvent(ofxDatGuiButtonEvent(btn));
    }

    pendingPresetLoad = false;
}

//--------------------------------------------------------------
void ofApp::updateMeshLoader()
{
    if (vertexLoader.isLoading() || meshBuilt) return;

    auto results = vertexLoader.getResults();

    clearAllAudio();

    for (const auto& [idx, player] : results)
    {
        if (player)
        {
            mVertexMappings[idx].player = player;
            ofLogNotice() << "Assigned player to vertex " << idx;
        }
        else
        {
            ofLogWarning() << "Failed to load player for vertex " << idx;
        }
    }

    sphere->buildMeshWithInputs(mVertexMappings);
    meshBuilt = true;
}

//--------------------------------------------------------------    
void ofApp::notifyMeshReady()
{
    if (!meshBuilt || loadingFinishedPrinted) return;

    std::cout << "Toggle audio by pressing the key 'o'" << std::endl
        << "Toggle triggers by pressing the key 't'" << std::endl
        << "Listening for incoming messages at PORT 9004" << std::endl;

    loadingFinishedPrinted = true;
}

//--------------------------------------------------------------
void ofApp::detectJointCollisions(const std::vector<glm::vec3>& pJointPositions)
{
    mCollisionIndices.clear();

    const auto& referencePoints = sphere->getUniqueVertices();

    for (std::size_t i = 0; i < referencePoints.size(); ++i)
    {
        const glm::vec3& spherePoint = referencePoints[i];

        auto it = std::find_if(mVertexMappings.begin(), mVertexMappings.end(),
            [i](const VertexMapping& vm) {
                return vm.vertexIndex == i;
            });

        if (it == mVertexMappings.end()) continue;

        float scale = ofToFloat(it->guiInput->getText());
        glm::vec3 scaledSpherePoint = glm::normalize(spherePoint) * scale;

        for (const auto& jointPoint : pJointPositions)
        {
            float dist = glm::length(jointPoint - scaledSpherePoint);
            if (dist <= MIN_COL_DISTANCE)
            {
                mCollisionIndices.push_back(i);
                break;
            }
        }
    }
}

//--------------------------------------------------------------
void ofApp::update()
{
    updatePresetTrigger();
    updateMeshLoader();
    notifyMeshReady();
    inst_GUI->update();
    updateOsc();
    updateOscSender();
}

//--------------------------------------------------------------
void ofApp::drawSphereMesh()
{
    glPointSize(5.0);
    sphere->getMesh().draw();
}

//--------------------------------------------------------------
void ofApp::drawAudioTrigger(const std::vector<std::size_t>& collisionIndices)
{
    for (auto it = collisionIndices.rbegin(); it != collisionIndices.rend(); ++it)
    {
        std::size_t vertexIdx = *it;

        if (vertexIdx < mVertexMappings.size())
        {
            const auto& mapping = mVertexMappings[vertexIdx];

            if (mapping.guiInput)
            {
                float scale = ofToFloat(mapping.guiInput->getText());
                glm::vec3 pos = glm::normalize(mapping.position) * scale;

                ofPushStyle();
                ofSetColor(0, 255, 0, 100);
                ofDrawSphere(pos, 5.0f);
                ofPopStyle();
            }
        }
    }
}

//--------------------------------------------------------------
void ofApp::drawCollisionPoints()
{
    if (mAudioThreadSync.load(std::memory_order_acquire))
    {
        mThreadedCollisionIndices = mCollisionIndices;
        mAudioThreadSync.store(false, std::memory_order_release);
    }

    const auto& indices = TEST_TRIGGER ? mManualCollisionIndices : mThreadedCollisionIndices;
    drawAudioTrigger(indices);
}

//--------------------------------------------------------------
void ofApp::drawImuSensor()
{
    glm::vec3 sensorPos = mArduinoQuat * glm::vec3(0, 0, -100);

    ofPushStyle();
    ofSetColor(255, 0, 0);
    ofDrawSphere(sensorPos, 5.0f);
    ofPopStyle();
}

//--------------------------------------------------------------
void ofApp::drawBoidPositions()
{
    ofPushStyle();
    ofSetColor(255, 0, 0);
    for (const auto& boidPos : mBoidPositions)
    {
        ofDrawSphere(boidPos, 1.0);
    }
    ofPopStyle();
}

//--------------------------------------------------------------
void ofApp::drawMocapJoints()
{
    ofPushStyle();
    ofSetColor(0, 255, 255);
    for (const auto& jointPos : mJointPositions)
    {
        ofDrawSphere(jointPos, 1.0);
    }
    ofPopStyle();
}

//--------------------------------------------------------------
void ofApp::draw()
{
    ofEnableDepthTest();

    cam.begin();

    drawSphereMesh();
    drawCollisionPoints();

    if (USE_IMU_SENSOR) drawImuSensor();
    if (USE_BOID_COLLISION) drawBoidPositions();
    if (USE_MOCAP_DATA) drawMocapJoints();

    cam.end();

    ofDisableDepthTest();

    if (mesh_GUI) {
        mesh_GUI->getFboTexture().draw(0, 0);
    }
}

//--------------------------------------------------------------
void ofApp::triggerAtIndexOrEnd(const std::vector<std::size_t>& indices)
{
    for (std::size_t colIndex : indices)
    {
        auto it = mVertexMap.find(colIndex);
        if (it == mVertexMap.end())
        {
            ofLogWarning() << "No mapping for vertex index " << colIndex;
            continue;
        }

        VertexMapping& vm = mVertexMappings[it->second];  // Get reference via index
        if (!vm.player)
        {
            ofLogWarning() << "No valid player for vertex index " << colIndex;
            continue;
        }

        auto& player = vm.player;

        float playbackSpeed = ofRandom(PITCH_RAND_MIN, PITCH_RAND_MAX);
        auto [grainStartNorm, grainEndNorm] = getRandomizedGrainBounds(SAMPLE_POS_START, SAMPLE_POS_END, RAND_POS_STRENGTH);

        bool isNewCollision = std::find(mCollisionIndicesBefore.begin(), mCollisionIndicesBefore.end(), colIndex) == mCollisionIndicesBefore.end();

        if (RETRIGGER_AT_INDEX && isNewCollision)
        {
            player->start(playbackSpeed, grainStartNorm, grainEndNorm);
        }

        if (RETRIGGER_AT_END && !player->isPlaying())
        {
            player->start(playbackSpeed, grainStartNorm, grainEndNorm);
        }
    }

    mCollisionIndicesBefore = indices;
}

//--------------------------------------------------------------
void ofApp::triggerWithMetronome(const std::vector<std::size_t>& indices)
{
    float jitterAmount = METRO_FREQUENCY * std::clamp(METRO_JITTER, 0.0f, 1.0f);
    float minFreq = std::max(0.1f, METRO_FREQUENCY - jitterAmount);
    float maxFreq = std::min(20000.0f, METRO_FREQUENCY + jitterAmount);
    float jitteredFreq = ofRandom(minFreq, maxFreq);

    mAudioPlayerMetro.setFrequency(jitteredFreq);

    if (mAudioPlayerMetro.tick())
    {
        mAudioGrains.erase(
            std::remove_if(mAudioGrains.begin(), mAudioGrains.end(),
                [](const Grain& grain) {
                    return !grain.player || !grain.player->isPlaying();
                }),
            mAudioGrains.end()
        );

        for (std::size_t colIndex : indices)
        {
            if (mAudioGrains.size() >= MAX_GRAINS)
                break;

            if (auto grain = createGrainFromVertex(colIndex)) {
                mAudioGrains.push_back(*grain);
            }
        }

        if (GRAIN_SHUFFLE)
        {
            std::shuffle(mAudioGrains.begin(), mAudioGrains.end(), std::default_random_engine{});
        }
    }
}

//--------------------------------------------------------------
std::shared_ptr<ofApp::Grain> ofApp::createGrainFromVertex(std::size_t index)
{
    auto it = mVertexMap.find(index);
    if (it == mVertexMap.end())
    {
        ofLogWarning() << "[createGrainFromVertex] No mapping for index " << index;
        return nullptr;
    }

    VertexMapping& vm = mVertexMappings[it->second];
    if (!vm.player)
    {
        ofLogWarning() << "[createGrainFromVertex] No player at index " << index;
        return nullptr;
    }

    auto grain = std::make_shared<Grain>();
    grain->player = std::make_shared<sfPlayer>(*(vm.player)); // deep copy
    grain->position = vm.position;

    float playbackSpeed = ofRandom(PITCH_RAND_MIN, PITCH_RAND_MAX);
    auto [grainStartNorm, grainEndNorm] = getRandomizedGrainBounds(SAMPLE_POS_START, SAMPLE_POS_END, RAND_POS_STRENGTH);
    grain->player->start(playbackSpeed, grainStartNorm, grainEndNorm);

    return grain;
}

//--------------------------------------------------------------
void ofApp::triggerAudio(std::vector<std::size_t> pCollisionIndices)
{
    if ((RETRIGGER_AT_INDEX || RETRIGGER_AT_END) && !METRONOME_TRIGGER)
    {
        triggerAtIndexOrEnd(pCollisionIndices);
    }

    if (METRONOME_TRIGGER)
    {
        triggerWithMetronome(pCollisionIndices);
    }
}

//--------------------------------------------------------------
void ofApp::audioOut(ofSoundBuffer& outBuffer)
{
    outBuffer.set(0.0f);

    updateOsc();

    if (USE_IMU_SENSOR)
    {
        glm::vec3 pSensorPosition = mArduinoQuat * glm::vec3(0, 0, -100);
        std::vector<glm::vec3> sensorPoints = { pSensorPosition };

        detectJointCollisions(sensorPoints);
    }

    if (USE_BOID_COLLISION)
        detectJointCollisions(mBoidPositions);

    if (USE_MOCAP_DATA)
        detectJointCollisions(mJointPositions);

    int numCh = outBuffer.getNumChannels();

    for (std::size_t buffer_count = 0; buffer_count < outBuffer.getNumFrames(); ++buffer_count)
    {
        std::array<double, 7> ambiSum = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

        if (METRONOME_TRIGGER)
        {
            for (auto& grain : mAudioGrains)
            {
                if (grain.player && grain.player->isPlaying())
                {
                    float sample = grain.player->play(inst_WindowType);

                    double azimuthRad = atan2(grain.position.y, grain.position.x);
                    double azimuthDeg = ofRadToDeg(azimuthRad);
                    if (azimuthDeg < 0) azimuthDeg += 360.0;

                    double distance = glm::length(grain.position);
                    if (distance < 0.0001) distance = 0.0001;

                    auto encoded = mAudioAmbiEncoder.play(sample, azimuthDeg, distance);

                    for (int i = 0; i < 7; ++i)
                        ambiSum[i] += encoded[i];
                }
            }
        }
        else if (RETRIGGER_AT_INDEX || RETRIGGER_AT_END)
        {
            for (const auto& vm : mVertexMappings)
            {
                if (vm.player && vm.player->isPlaying())
                {
                    float sample = vm.player->play(inst_WindowType);

                    double azimuthRad = atan2(vm.position.y, vm.position.x);
                    double azimuthDeg = ofRadToDeg(azimuthRad);
                    if (azimuthDeg < 0) azimuthDeg += 360.0;

                    double distance = glm::length(vm.position);
                    if (distance < 0.0001) distance = 0.0001;

                    auto encoded = mAudioAmbiEncoder.play(sample, azimuthDeg, distance);

                    for (int i = 0; i < 7; ++i)
                        ambiSum[i] += encoded[i];
                }
            }
        }

        if (meshBuilt && !TEST_TRIGGER && mAudioTrigger)
            triggerAudio(mCollisionIndices);

        if (TEST_TRIGGER)
            triggerAudio(mManualCollisionIndices);

        if (mAudioRecording)
        {
            // If you want to record the mixed sum, you need to decode to stereo first, or record ambisonic channels
            // For example, just recording sum of W channel for now
            writeFileToBuffer(ambiSum[0]);
        }

        static std::vector<int> channelAngles_2ch = { 45, 270 };
        static std::vector<int> channelAngles_4ch = { 45, 135, 225, 315 };
        static std::vector<int> channelAngles_6ch = { 60, 120, 180, 240, 300, 360 };
        static std::vector<int> channelAngles_8ch = { 45, 90, 135, 180, 225, 270, 315, 360 };

        const std::vector<int>* channelAngles = nullptr;

        switch (numCh)
        {
        case 2: channelAngles = &channelAngles_2ch; break;
        case 4: channelAngles = &channelAngles_4ch; break;
        case 6: channelAngles = &channelAngles_6ch; break;
        case 8: channelAngles = &channelAngles_8ch; break;
        default:
            ofLogWarning() << "Unsupported channel count: " << numCh;
            return;
        }

        for (int ch = 0; ch < numCh; ++ch)
        {
            float speakerAzimuthRad = PI * (*channelAngles)[ch] / 180.0f;
            double decodedSample = mAudioAmbiDecoder[ch].play(ambiSum, speakerAzimuthRad);
            outBuffer[buffer_count * numCh + ch] = static_cast<float>(decodedSample);
        }
    }

    mAudioThreadSync.store(true, std::memory_order_release);
}

//--------------------------------------------------------------
void ofApp::clearAllAudio()
{
    for (auto& vm : mVertexMappings)
    {
        if (vm.player)
            vm.player->stop();
    }

    for (auto& grain : mAudioGrains)
    {
        if (grain.player)
            grain.player->stop();
    }

    mAudioGrains.clear();
}

//--------------------------------------------------------------
void ofApp::writeFileSetup(const std::string& pSfName)
{
    writeFileSampleBuffer.clear();

    soundFileBuffer = SndfileHandle(ofToDataPath(pSfName), SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_DOUBLE, 1, 48000);
}

//--------------------------------------------------------------
void ofApp::writeFileToBuffer(const double& pInput)
{
    writeFileSampleBuffer.push_back(pInput);
}

//--------------------------------------------------------------
void ofApp::writeFileToDisk(const std::vector<double>& pSampleBuffer)
{
    soundFileBuffer.write(pSampleBuffer.data(), pSampleBuffer.size());
}

//--------------------------------------------------------------
void ofApp::loadSoundFolderByIndex(int index, bool forceReload, int xRes, int yRes)
{
    if (index < 0 || index >= availableAudioFolders.size()) 
    {
        ofLogWarning() << "[ loadSoundFolderByIndex ] Invalid index: " << index;
        return;
    }

    if (!forceReload && index == currentLoadedFolderIndex) 
    {
        ofLogNotice() << "[ loadSoundFolderByIndex ] Folder index " << index << " is already loaded.";
        return;
    }

    std::string folderName = availableAudioFolders[index];
    std::string folderPath = "data/" + folderName;
    std::vector<std::string> filePaths = directoryIterator(folderPath);

    if (mVertexMappings.size() > filePaths.size())
    {
        ofLogWarning() << "[ loadSoundFolderByIndex ]" << filePaths.size() << " soundfiles & " << mVertexMappings.size() << " vertices in Folder : " << index;
        return;
    }

    if (filePaths.empty()) {
        ofLogWarning() << "[ loadSoundFolderByIndex ] Folder is empty: " << folderPath;
        return;
    }

    std::vector<VertexLoadTask> tasks;
    for (std::size_t i = 0; i < mVertexMappings.size(); ++i) {
        if (i < filePaths.size()) {
            std::string relPath = folderName + "/" + filePaths[i];
            tasks.push_back({ static_cast<int>(i), relPath });
        }
    }

    vertexLoader.start(tasks);
    meshBuilt = false;
    currentLoadedFolderIndex = index;

    ofLogNotice() << "[loadSoundFolderByIndex] Loaded folder: " << folderName;
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key)
{
    if (key == 't')
    {
        mAudioTrigger = !mAudioTrigger;

        if (mAudioTrigger) 
        {
            clearAllAudio();
        }
        std::cout << "Audio trigger: " << mAudioTrigger << "\n";
    }


    if (key == 'o')
    {
        mAudioSoundStreamOnOffToggle = (mAudioSoundStreamOnOffToggle + 1) % 2;
        if (mAudioSoundStreamOnOffToggle == 0)
        {
            mSoundStream.close();
            std::cout << "Audio engine close" << std::endl;
        }
        if (mAudioSoundStreamOnOffToggle == 1)
        {
            startAudioBackend();
            std::cout << "Audio engine start" << std::endl;
        }
    }

    if (key == 'r')
    {
        mAudioRecording = true;
        writeFileSetup("audioTest.wav");
        std::cout << "Recording audio" << std::endl;
    }
    if (key == 'w' && mAudioRecording)
    {
        writeFileToDisk(writeFileSampleBuffer);
        std::cout << "Write audio recording to disc" << std::endl;
        mAudioRecording = false;
    }
    if (key == ' ')
    {
        colIndexTestCount = (colIndexTestCount += 1) % sphere->getUniqueVertices().size();
        mManualCollisionIndices = { colIndexTestCount };
        std::cout << colIndexTestCount << std::endl;
    }
    if (key == 'p')
    {
        RNN_MOTION_CONTINUATION = !RNN_MOTION_CONTINUATION;
    }

    float scrollStep = 10.0f;

    if (key == OF_KEY_UP) {
        mesh_GUI->scroll(-scrollStep);
    }
    else if (key == OF_KEY_DOWN) {
        mesh_GUI->scroll(scrollStep);
    }
}

//--------------------------------------------------------------
void ofApp::onSliderEvent(ofxDatGuiSliderEvent e)
{
    if (e.target->is(inst_GUI_FaderNames[0])) METRO_FREQUENCY = e.value;
    if (e.target->is(inst_GUI_FaderNames[1])) METRO_JITTER = e.value;
    if (e.target->is(inst_GUI_FaderNames[2])) PITCH_RAND_MIN = e.value;
    if (e.target->is(inst_GUI_FaderNames[3])) PITCH_RAND_MAX = e.value;
    if (e.target->is(inst_GUI_FaderNames[4])) SAMPLE_POS_START = e.value;
    if (e.target->is(inst_GUI_FaderNames[5])) SAMPLE_POS_END = e.value;
    if (e.target->is(inst_GUI_FaderNames[6])) RAND_POS_STRENGTH = e.value;
    if (e.target->is(inst_GUI_FaderNames[7])) MIN_COL_DISTANCE = e.value;
}

//--------------------------------------------------------------
void ofApp::onDropdownEvent(ofxDatGuiDropdownEvent e)
{
    if (e.target->is("SELECT_WINDOW_FUNCTION"))
    {
        inst_GUI_SelectWindow = e.child;

        inst_WindowType = static_cast<WindowFunction::Type>(e.child);

        std::cout << "Selected window type index: " << inst_GUI_SelectWindow << std::endl;
    }

    if (e.target->is("SELECT_SOUND_FOLDER"))
    {
        std::string selectedFolder = e.target->getChildAt(e.child)->getName();
        std::string folderPath = "data/" + selectedFolder;

        std::vector<std::string> fileNames = directoryIterator(folderPath);

        if (fileNames.empty())
        {
            ofLogWarning() << "[Dropdown] Folder is empty: " << folderPath;
            return;
        }

        std::vector<VertexLoadTask> tasks;
        for (size_t i = 0; i < mVertexMappings.size(); ++i) {
            if (i < fileNames.size()) {
                tasks.push_back({ static_cast<int>(i), selectedFolder + "/" + fileNames[i] });
            }
        }

        vertexLoader.start(tasks);
        meshBuilt = false;
        ofLogNotice() << "Started loading from folder: " << folderPath << " with " << tasks.size() << " files";
    }
}

//--------------------------------------------------------------
void ofApp::onToggleEvent(ofxDatGuiToggleEvent e)
{
    if (e.target->is("RETRIGGER_AT_INDEX")) RETRIGGER_AT_INDEX = e.checked;
    if (e.target->is("RETRIGGER_AT_END")) RETRIGGER_AT_END = e.checked;
    if (e.target->is("METRONOME_TRIGGER")) METRONOME_TRIGGER = e.checked;
    if (e.target->is("GRAIN_SHUFFLE")) GRAIN_SHUFFLE = e.checked;
    if (e.target->is("UNLOCK_PRESET")) mGuiUnlockPresets = e.checked;
}

//--------------------------------------------------------------
void ofApp::onTextInputEvent(ofxDatGuiTextInputEvent e)
{
    if (e.target->is("SEL_PRST_NR")) {
        mGuiSelectPresetNumber = e.text;
        return;
    }

    if (e.target->is("SELECT_FOLDER")) {
        try {
            int index = std::stoi(e.text);
            loadSoundFolderByIndex(index, false, currentXRes, currentYRes);
        }
        catch (const std::exception& ex) {
            ofLogWarning() << "[onTextInputEvent] Invalid index input: " << e.text;
        }
        return;
    }

    sphere->buildMeshWithInputs(mVertexMappings);
}


//--------------------------------------------------------------
void ofApp::onButtonEvent(ofxDatGuiButtonEvent e)
{
    if (e.target->is("READ_WRITE_PRST")) mGuiReadOrWritePreset = e.target;

    if (mGuiUnlockPresets && mGuiReadOrWritePreset)
    {
        saveGuiValuesAsFile(mGuiSelectPresetNumber);
    }
    else if (!mGuiUnlockPresets && mGuiReadOrWritePreset)
    {
        readGuiValuesFromFile(mGuiSelectPresetNumber);
    }
}

//--------------------------------------------------------------
void ofApp::saveGuiValuesAsFile(const std::string& presetNumber)
{
    ofJson json;

    for (const auto& name : inst_GUI_FaderNames)
    {
        json["inst_GUI"][name] = inst_GUI->getSlider(name)->getValue();
    }

    json["inst_GUI"]["RETRIGGER_AT_INDEX"] = inst_GUI->getToggle("RETRIGGER_AT_INDEX")->getChecked();
    json["inst_GUI"]["RETRIGGER_AT_END"] = inst_GUI->getToggle("RETRIGGER_AT_END")->getChecked();
    json["inst_GUI"]["METRONOME_TRIGGER"] = inst_GUI->getToggle("METRONOME_TRIGGER")->getChecked();
    json["inst_GUI"]["GRAIN_SHUFFLE"] = inst_GUI->getToggle("GRAIN_SHUFFLE")->getChecked();

    json["inst_GUI"]["SELECT_WINDOW_FUNCTION"] = inst_GUI->getDropdown("SELECT_WINDOW_FUNCTION")->getSelected()->getLabel();

    json["inst_GUI"]["SELECT_FOLDER"] = inst_GUI->getTextInput("SELECT_FOLDER")->getText();

    json["inst_GUI"]["X_RES"] = inst_GUI->getTextInput("X_RES")->getText();
    json["inst_GUI"]["Y_RES"] = inst_GUI->getTextInput("Y_RES")->getText();

    for (size_t i = 0; i < mVertexMappings.size(); ++i)
    {
        json["mesh_GUI"]["Vertex_" + ofToString(i)] = mVertexMappings[i].guiInput->getText();
    }

    std::string fileName = "preset_number_" + presetNumber + ".json";
    ofSaveJson(fileName, json);
}

//--------------------------------------------------------------
void ofApp::readGuiValuesFromFile(const std::string& presetNumber)
{
    std::string fileName = "preset_number_" + presetNumber + ".json";

    if (!ofFile::doesFileExist(fileName)) return;

    ofJson json = ofLoadJson(fileName);

    bool resolutionChanged = false;
    bool folderChanged = false;
    int newXRes = currentXRes;
    int newYRes = currentYRes;
    int newFolderIndex = currentLoadedFolderIndex;

    for (const auto& name : inst_GUI_FaderNames)
    {
        if (json["inst_GUI"].contains(name))
        {
            float val = json["inst_GUI"][name].get<float>();
            inst_GUI->getSlider(name)->setValue(val);

            if (name == inst_GUI_FaderNames[0]) METRO_FREQUENCY = val;
            else if (name == inst_GUI_FaderNames[1]) METRO_JITTER = val;
            else if (name == inst_GUI_FaderNames[2]) PITCH_RAND_MIN = val;
            else if (name == inst_GUI_FaderNames[3]) PITCH_RAND_MAX = val;
            else if (name == inst_GUI_FaderNames[4]) SAMPLE_POS_START = val;
            else if (name == inst_GUI_FaderNames[5]) SAMPLE_POS_END = val;
            else if (name == inst_GUI_FaderNames[6]) RAND_POS_STRENGTH = val;
            else if (name == inst_GUI_FaderNames[7]) MIN_COL_DISTANCE = val;
        }
    }

    auto loadToggle = [&](const std::string& key)
        {
            if (json["inst_GUI"].contains(key))
            {
                bool checked = json["inst_GUI"][key].get<bool>();
                inst_GUI->getToggle(key)->setChecked(checked);

                if (key == "RETRIGGER_AT_INDEX")  RETRIGGER_AT_INDEX = checked;
                else if (key == "RETRIGGER_AT_END") RETRIGGER_AT_END = checked;
                else if (key == "METRONOME_TRIGGER") METRONOME_TRIGGER = checked;
                else if (key == "GRAIN_SHUFFLE") GRAIN_SHUFFLE = checked;
            }
        };

    loadToggle("RETRIGGER_AT_INDEX");
    loadToggle("RETRIGGER_AT_END");
    loadToggle("METRONOME_TRIGGER");
    loadToggle("GRAIN_SHUFFLE");

    if (json["inst_GUI"].contains("SELECT_WINDOW_FUNCTION"))
    {
        std::string label = json["inst_GUI"]["SELECT_WINDOW_FUNCTION"];
        auto dropdown = inst_GUI->getDropdown("SELECT_WINDOW_FUNCTION");

        if (dropdown) {
            int n = dropdown->size();
            bool found = false;

            for (int i = 0; i < n; ++i)
            {
                auto option = dropdown->getChildAt(i);
                if (option && option->getLabel() == label)
                {
                    dropdown->select(i);
                    inst_GUI_SelectWindow = i;
                    inst_WindowType = static_cast<WindowFunction::Type>(i);
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                ofLogWarning() << "[SELECT_WINDOW_FUNCTION] Label not found: " << label;
            }
        }
        else
        {
            ofLogWarning() << "[SELECT_WINDOW_FUNCTION] Dropdown not found.";
        }
    }

    if (json["inst_GUI"].contains("X_RES") && json["inst_GUI"].contains("Y_RES"))
    {
        std::string xResInput = json["inst_GUI"]["X_RES"].get<std::string>();
        std::string yResInput = json["inst_GUI"]["Y_RES"].get<std::string>();
        inst_GUI->getTextInput("X_RES")->setText(xResInput);
        inst_GUI->getTextInput("Y_RES")->setText(yResInput);

        newXRes = ofToInt(xResInput);
        newYRes = ofToInt(yResInput);

        if (newXRes != currentXRes || newYRes != currentYRes)
        {
            rebuildSphere(newXRes, newYRes);
            resolutionChanged = true;
        }
    }

    for (size_t i = 0; i < mVertexMappings.size(); ++i)
    {
        std::string key = "Vertex_" + ofToString(i);
        if (json["mesh_GUI"].contains(key))
        {
            std::string stringOffset = json["mesh_GUI"][key];
            mVertexMappings[i].guiInput->setText(stringOffset);
            sphere->setOffset(i, std::stof(stringOffset));
        }
    }

    sphere->buildMeshWithInputs(mVertexMappings);

    if (json["inst_GUI"].contains("SELECT_FOLDER"))
    {
        std::string folderText = json["inst_GUI"]["SELECT_FOLDER"];
        inst_GUI->getTextInput("SELECT_FOLDER")->setText(folderText);

        try {
            newFolderIndex = std::stoi(folderText);
            if (newFolderIndex != currentLoadedFolderIndex) {
                folderChanged = true;
            }
        }
        catch (const std::exception& ex) {
            ofLogWarning() << "[readGuiValuesFromFile] Invalid folder index in preset: " << folderText;
            newFolderIndex = -1;
        }
    }
    else
    {
        ofLogNotice() << "[readGuiValuesFromFile] No SELECT_FOLDER found in preset.";
    }

    forceFileReload = resolutionChanged || folderChanged;

    if (newFolderIndex >= 0) {
        loadSoundFolderByIndex(newFolderIndex, forceFileReload, newXRes, newYRes);
        currentLoadedFolderIndex = newFolderIndex;
        currentXRes = newXRes;
        currentYRes = newYRes;
    }

    forceFileReload = false;
}


//--------------------------------------------------------------
void ofApp::exit()
{
}

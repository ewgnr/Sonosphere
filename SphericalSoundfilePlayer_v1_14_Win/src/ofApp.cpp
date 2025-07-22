#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup()
{
    ofSetLogLevel(OF_LOG_VERBOSE);
    ofSetLoggerChannel(std::make_shared<ofConsoleLoggerChannel>());

    ofBackground(0);

    selectAudioDevice();

    sphere = std::make_unique<uvSphere>(10, 10, 1.0f);

    mAudioGrains.reserve(MAX_GRAINS);
    mAudioAmbiDecoder.resize(mAudioNumOutputChannels);

    ofSetVerticalSync(true);

    mesh_GUI = new ofxDatGui(ofxDatGuiAnchor::TOP_RIGHT);
    mesh_GUI->enableFboMode(true, ofGetWidth(), ofGetHeight());
    mesh_GUI->setWidth(180, 0.6f);
    mesh_GUI->onTextInputEvent(this, &ofApp::onTextInputEvent);

    const auto& vertices = sphere->getUniqueVertices();
    std::size_t vertexCount = vertices.size();

    inst_GUI = new ofxDatGui(ofxDatGuiAnchor::TOP_LEFT);
    inst_GUI->addSlider(inst_GUI_FaderNames[0], 0.001, 20.0, 1.0)->setPrecision(7);
    inst_GUI->addSlider(inst_GUI_FaderNames[1], 0.001, 12.0, 1.0)->setPrecision(7);
    inst_GUI->addSlider(inst_GUI_FaderNames[2], 0.001, 12.0, 1.0)->setPrecision(7);
    inst_GUI->addSlider(inst_GUI_FaderNames[3], 0.0, 1.0, 0.0)->setPrecision(7);
    inst_GUI->addSlider(inst_GUI_FaderNames[4], 0.0, 1.0, 1.0)->setPrecision(7);
    inst_GUI->addSlider(inst_GUI_FaderNames[5], 1.0, 100.0, 50.0)->setPrecision(7);

    inst_GUI->addToggle("RETRIGGER_AT_INDEX", true);
    inst_GUI->addToggle("RETRIGGER_AT_END", false);
    inst_GUI->addToggle("METRONOME_TRIGGER", false);
    inst_GUI->addToggle("GRAIN_SHUFFLE", false);
    inst_GUI->addDropdown("SELECT_WINDOW_FUNCTION", windows);
    inst_GUI->addTextInput("SELECT_FOLDER", "0")->setInputType(ofxDatGuiInputType::NUMERIC);
    inst_GUI->addTextInput("X_RES", "10")->setInputType(ofxDatGuiInputType::NUMERIC);
    inst_GUI->addTextInput("Y_RES", "10")->setInputType(ofxDatGuiInputType::NUMERIC);

    inst_GUI->onSliderEvent(this, &ofApp::onSliderEvent);
    inst_GUI->onDropdownEvent(this, &ofApp::onDropdownEvent);
    inst_GUI->onToggleEvent(this, &ofApp::onToggleEvent);
    inst_GUI->onButtonEvent(this, &ofApp::onButtonEvent);
    inst_GUI->onTextInputEvent(this, &ofApp::onTextInputEvent);

    presetControl_GUI = new ofxDatGui(ofxDatGuiAnchor::BOTTOM_LEFT);
    presetControl_GUI->addToggle("UNLOCK_PRESET");
    presetControl_GUI->addButton("READ_WRITE_PRST");
    presetControl_GUI->addTextInput("SEL_PRST_NR")->setInputType(ofxDatGuiInputType::NUMERIC);

    presetControl_GUI->onToggleEvent(this, &ofApp::onToggleEvent);
    presetControl_GUI->onButtonEvent(this, &ofApp::onButtonEvent);
    presetControl_GUI->onTextInputEvent(this, &ofApp::onTextInputEvent);

    availableAudioFolders = getFolderNames();

    if (availableAudioFolders.empty()) {
        ofLogWarning() << "[setup] No audio folders found inside data/";
        return;
    }

    mVertexMappings.clear();
    for (std::size_t vertexIdx = 0; vertexIdx < vertexCount; ++vertexIdx)
    {
        VertexMapping vm;
        vm.guiIndex = vertexIdx;
        vm.vertexIndex = vertexIdx;
        vm.position = vertices[vertexIdx];

        auto input = mesh_GUI->addTextInput("V " + ofToString(vertexIdx), "100.0");
        input->setInputType(ofxDatGuiInputType::NUMERIC);
        vm.guiInput = input;

        mVertexMappings.push_back(vm);
    }

    cam.setDistance(500);
    cam.removeAllInteractions();
    cam.addInteraction(ofEasyCam::TRANSFORM_ROTATE, OF_MOUSE_BUTTON_RIGHT);
    cam.setLensOffset(ofVec2f(-0.3, 0));

    setupOsc();

    loadSoundFolderByIndex(0, forceFileReload, currentXRes, currentYRes);
}

//--------------------------------------------------------------
void ofApp::rebuildSphere(int xRes, int yRes)
{
    vertexLoader.stop(); // stop async loading safely before deleting GUI

    sphere = std::make_unique<uvSphere>(xRes, yRes, 1.0f);

    delete mesh_GUI;
    mesh_GUI = new ofxDatGui(ofxDatGuiAnchor::TOP_RIGHT);
    mesh_GUI->enableFboMode(true, ofGetWidth(), ofGetHeight());
    mesh_GUI->setWidth(180, 0.6f);
    mesh_GUI->onTextInputEvent(this, &ofApp::onTextInputEvent);

    mVertexMappings.clear();
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
    }

    ofLogNotice() << "[rebuildSphere] Rebuilt with " << vertices.size() << " vertices.";
}

//--------------------------------------------------------------
void ofApp::selectAudioDevice()
{
    mSettings.setApi(ofSoundDevice::Api::MS_DS);
    mDevices = mSoundStream.getDeviceList(ofSoundDevice::Api::MS_DS);

    for (int i = 0; i < mDevices.size(); ++i)
    {
        std::cout << mDevices[i] << "\n";
    }

    std::cout << "select imput device by index" << std::endl;
    std::cin >> selInputDevice;
}

//--------------------------------------------------------------
void ofApp::startAudioBackend()
{
    mSettings.setOutDevice(mDevices[std::stoi(selInputDevice)]);
    mSettings.setInDevice(mDevices[std::stoi(selInputDevice)]);
    mSettings.numOutputChannels = mAudioNumOutputChannels;
    mSettings.numInputChannels = 2;
    mSettings.sampleRate = mAudioSampleRate;
    mSettings.bufferSize = mAudioBufferSize;
    mSettings.numBuffers = 4;
    mSettings.setOutListener(this);
    mSettings.setInListener(this);
    mSoundStream.setup(mSettings);
}

//--------------------------------------------------------------
void ofApp::setupOsc()
{
    try
    {
        mSelf = std::shared_ptr<ofApp>(this);
        // Motion Capturing Input 9004
        mOscReceiver = new dab::OscReceiver("MocapReceiver", 9004);
        mOscReceiver->registerOscListener(std::weak_ptr<ofApp>(mSelf));
        mOscReceiver->start();

        mOscSendAddress = "127.0.0.1";
        mOscSendPort = 9010;
        mOscSender = new dab::OscSender("MocapSender", mOscSendAddress, mOscSendPort);
    }
    catch (dab::Exception& e)
    {
        std::cout << e << "\n";
    }
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
void ofApp::updateOsc()
{
    mOscLock.lock();

    while (mOscMessageQueue.size() > 0)
    {
        std::shared_ptr< dab::OscMessage > oscMessage = mOscMessageQueue[0];

        updateOsc(oscMessage);

        mOscMessageQueue.pop_front();
    }

    mOscLock.unlock();
}

//--------------------------------------------------------------
void ofApp::updateOsc(std::shared_ptr<dab::OscMessage> pMessage)
{
    try
    {
        std::string address = pMessage->address();

        const std::vector<dab::_OscArg*>& arguments = pMessage->arguments();

        if (address == "/mocap/joint/pos_world" || address == "/mocap/0/joint/pos_world")
        {
            size_t jointCount = mJointPositions.size();
            size_t argCount = arguments.size();

            jointCount = std::min(jointCount, argCount / 3);

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
            }
        }
    }

    catch (dab::Exception& e)
    {
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
void ofApp::triggerAudio(std::vector<std::size_t> pCollisionIndices)
{
    float playbackSpeed = ofRandom(PITCH_RAND_MIN, PITCH_RAND_MAX);

    if ((RETRIGGER_AT_INDEX || RETRIGGER_AT_END) && !METRONOME_TRIGGER)
    {
        for (std::size_t colIndex : pCollisionIndices)
        {
            auto it = std::find_if(mVertexMappings.begin(), mVertexMappings.end(),
                [colIndex](const VertexMapping& vm) {
                    return vm.vertexIndex == colIndex;
                });

            if (it == mVertexMappings.end() || !it->player)
            {
                ofLogWarning() << "No valid player for vertex index " << colIndex;
                continue;
            }

            auto& player = it->player;

            bool isNewCollision = std::find(mCollisionIndicesBefore.begin(), mCollisionIndicesBefore.end(), colIndex) == mCollisionIndicesBefore.end();
            /*
            if (RETRIGGER_AT_INDEX) {
                player->start(playbackSpeed, SAMPLE_POS_START, SAMPLE_POS_END);
            }
            */
            if (RETRIGGER_AT_INDEX && isNewCollision)
            {
                player->start(playbackSpeed, SAMPLE_POS_START, SAMPLE_POS_END);
            }

            if (RETRIGGER_AT_END && !player->isPlaying())
            {
                player->start(playbackSpeed, SAMPLE_POS_START, SAMPLE_POS_END);
            }
        }

        mCollisionIndicesBefore = pCollisionIndices;
    }

    if (METRONOME_TRIGGER)
    {
        mAudioPlayerMetro.setFrequency(METRO_FREQUENCY);

        if (mAudioPlayerMetro.tick())
        {
            mAudioGrains.erase(
                std::remove_if(mAudioGrains.begin(), mAudioGrains.end(),
                    [](const Grain& grain) {
                        return !grain.player || !grain.player->isPlaying();
                    }),
                mAudioGrains.end()
                );

            for (std::size_t colIndex : pCollisionIndices)
            {
                if (mAudioGrains.size() >= MAX_GRAINS)
                    break;

                auto it = std::find_if(mVertexMappings.begin(), mVertexMappings.end(),
                    [colIndex](const VertexMapping& vm) {
                        return vm.vertexIndex == colIndex;
                    });

                if (it != mVertexMappings.end() && it->player)
                {
                    Grain grain;
                    grain.player = std::make_shared<sfPlayer>(*(it->player));
                    grain.position = it->position;

                    float playbackSpeed = ofRandom(PITCH_RAND_MIN, PITCH_RAND_MAX);
                    grain.player->start(playbackSpeed, SAMPLE_POS_START, SAMPLE_POS_END);
                    mAudioGrains.push_back(grain);
                }
                else
                {
                    ofLogWarning() << "No valid grain player for collision index " << colIndex;
                }
            }

            if (GRAIN_SHUFFLE)
            {
                std::shuffle(mAudioGrains.begin(), mAudioGrains.end(), std::default_random_engine{});
            }
        }
    }
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
void ofApp::update()
{
    if (!vertexLoader.isLoading() && !meshBuilt)
    {
        auto results = vertexLoader.getResults();

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

        sphere->buildMeshWithInputs(mVertexMappings);  // call mesh rebuild if needed
        meshBuilt = true;
    }

    if (meshBuilt && !loadingFinishedPrinted)
    {
        std::cout << "turn on audio by pressing the key 'o'" << std::endl;
        loadingFinishedPrinted = true;
    }
}

//--------------------------------------------------------------
void ofApp::draw()
{
    ofEnableDepthTest();

    cam.begin();

    glPointSize(5.0);
    sphere->getMesh().draw();

    if (mAudioThreadSync.load(std::memory_order_acquire))
    {
        mThreadedCollisionIndices = mCollisionIndices;
        mAudioThreadSync.store(false, std::memory_order_release);
    }

    if (TEST_TRIGGER)
    {
        drawAudioTrigger(mManualCollisionIndices);
    }
    else
    {
        drawAudioTrigger(mThreadedCollisionIndices);
    }

    ofPushStyle();
    ofSetColor(0, 255, 255);
    for (auto jointPos : mJointPositions)
    {
        ofDrawSphere(jointPos, 1.0);
    }
    ofPopStyle();
    cam.end();

    ofDisableDepthTest();

    updateOscSender();

    if (mesh_GUI) {
        mesh_GUI->getFboTexture().draw(0, 0);
    }
}

//--------------------------------------------------------------
void ofApp::audioOut(ofSoundBuffer& outBuffer)
{
    updateOsc();

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

    mAudioThreadSync.store(true, memory_order_release);
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
void ofApp::loadSoundFolderByIndex(int index, bool forceReload = false, int xRes = 10, int yRes = 10) 
{
    if (index < 0 || index >= availableAudioFolders.size()) 
    {
        ofLogWarning() << "[loadSoundFolderByIndex] Invalid index: " << index;
        return;
    }

    if (!forceReload && index == currentLoadedFolderIndex) 
    {
        ofLogNotice() << "[loadSoundFolderByIndex] Folder index " << index << " is already loaded.";
        return;
    }

    std::string folderName = availableAudioFolders[index];
    std::string folderPath = "data/" + folderName;
    std::vector<std::string> filePaths = directoryIterator(folderPath);

    if (mVertexMappings.size() > filePaths.size())
    {
        ofLogWarning() << "[loadSoundFolderByIndex]" << filePaths.size() << "soundfiles & " << mVertexMappings.size() << "vertices in Folder : " << index;
        return;
    }

    if (filePaths.empty()) {
        ofLogWarning() << "[loadSoundFolderByIndex] Folder is empty: " << folderPath;
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

        std::cout << "mAudioTrigger " << mAudioTrigger << "\n";
    }
    if (key == 'o')
    {
        mAudioSoundStreamOnOffToggle = (mAudioSoundStreamOnOffToggle + 1) % 2;
        if (mAudioSoundStreamOnOffToggle == 0)
        {
            mSoundStream.close();
            std::cout << "audio engine close" << std::endl;
        }
        if (mAudioSoundStreamOnOffToggle == 1)
        {
            startAudioBackend();;
            std::cout << "audio engine start" << std::endl;
        }
    }

    if (key == 'r')
    {
        mAudioRecording = true;
        writeFileSetup("audioTest.wav");
        std::cout << "recording audio" << std::endl;
    }
    if (key == 'w' && mAudioRecording)
    {
        writeFileToDisk(writeFileSampleBuffer);
        std::cout << "write audio recording to disc" << std::endl;
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
    if (e.target->is(inst_GUI_FaderNames[1])) PITCH_RAND_MIN = e.value;
    if (e.target->is(inst_GUI_FaderNames[2])) PITCH_RAND_MAX = e.value;
    if (e.target->is(inst_GUI_FaderNames[3])) SAMPLE_POS_START = e.value;
    if (e.target->is(inst_GUI_FaderNames[4])) SAMPLE_POS_END = e.value;
    if (e.target->is(inst_GUI_FaderNames[5])) MIN_COL_DISTANCE = e.value;
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

    for (const auto& name : inst_GUI_FaderNames)
    {
        if (json["inst_GUI"].contains(name))
        {
            float val = json["inst_GUI"][name].get<float>();
            inst_GUI->getSlider(name)->setValue(val);

            if (name == inst_GUI_FaderNames[0]) METRO_FREQUENCY = val;
            else if (name == inst_GUI_FaderNames[1]) PITCH_RAND_MIN = val;
            else if (name == inst_GUI_FaderNames[2]) PITCH_RAND_MAX = val;
            else if (name == inst_GUI_FaderNames[3]) SAMPLE_POS_START = val;
            else if (name == inst_GUI_FaderNames[4]) SAMPLE_POS_END = val;
            else if (name == inst_GUI_FaderNames[5]) MIN_COL_DISTANCE = val;
        }
    }

    if (json["inst_GUI"].contains("RETRIGGER_AT_INDEX"))
    {
        bool checked = json["inst_GUI"]["RETRIGGER_AT_INDEX"].get<bool>();
        inst_GUI->getToggle("RETRIGGER_AT_INDEX")->setChecked(checked);
        RETRIGGER_AT_INDEX = checked;
    }
    if (json["inst_GUI"].contains("RETRIGGER_AT_END"))
    {
        bool checked = json["inst_GUI"]["RETRIGGER_AT_END"].get<bool>();
        inst_GUI->getToggle("RETRIGGER_AT_END")->setChecked(checked);
        RETRIGGER_AT_END = checked;
    }
    if (json["inst_GUI"].contains("METRONOME_TRIGGER"))
    {
        bool checked = json["inst_GUI"]["METRONOME_TRIGGER"].get<bool>();
        inst_GUI->getToggle("METRONOME_TRIGGER")->setChecked(checked);
        METRONOME_TRIGGER = checked;
    }
    if (json["inst_GUI"].contains("GRAIN_SHUFFLE"))
    {
        bool checked = json["inst_GUI"]["GRAIN_SHUFFLE"].get<bool>();
        inst_GUI->getToggle("GRAIN_SHUFFLE")->setChecked(checked);
        GRAIN_SHUFFLE = checked;
    }
   
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

        int xRes = ofToInt(xResInput);
        int yRes = ofToInt(yResInput);

        if (xRes != currentXRes || yRes != currentYRes)
        {
            rebuildSphere(xRes, yRes);
            currentXRes = xRes;
            currentYRes = yRes;
            forceFileReload = true;
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
            int index = std::stoi(folderText);
            loadSoundFolderByIndex(index, forceFileReload, currentXRes, currentYRes);
        }
        catch (const std::exception& ex) {
            ofLogWarning() << "[readGuiValuesFromFile] Invalid folder index in preset: " << folderText;
        }
    }
    else
    {
        ofLogNotice() << "[readGuiValuesFromFile] No SELECT_FOLDER found in preset.";
    }

    forceFileReload = false;
}


//--------------------------------------------------------------
void ofApp::exit()
{
}
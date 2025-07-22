#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() 
{    
    // select audio device
    selectAudioDevice();

    // load audio files
    std::vector<string> tempFilePath;
    tempFilePath = directoryIterator(ofToDataPath("audio/"));

    for (size_t sfPlayerIndex = 0; sfPlayerIndex < NUMBER_OF_SOUNDFILES; sfPlayerIndex++)
    {
        mAudioPlayers[sfPlayerIndex].read(ofToDataPath("audio/" + tempFilePath[sfPlayerIndex], true));
    }
    
    // graphic setup for 3d renderer
    ofSetVerticalSync(true);

    ofSetDrawBitmapMode(OF_BITMAPMODE_MODEL);

    // camera setup
    mVisCam.setPosition(ofGetWidth() * 0.7, ofGetHeight() * 0.5, ofGetHeight());
    glm::vec3 targetPosition = { ofGetWidth() * 0.5, ofGetHeight() * 0.5, 0 };
    mVisCam.setTarget(targetPosition);
    mVisCam.removeAllInteractions();
    mVisCam.addInteraction(ofEasyCam::TRANSFORM_ROTATE, OF_MOUSE_BUTTON_RIGHT);
    mVisCam.setLensOffset(ofVec2f(-0.4, 0));

    // sphere setup
    mVisSphere.setup(ROWS, COLS, RADIUS);
    mVisSphere.verticies();
    mVisSphere.mesh();

    std::cout << "indiciesString " << mVisSphere.indiciesString.size() << std::endl;
    std::cout << "refPoint " << mVisSphere.ref_vert_pnt.size() << std::endl;

    calculate2dDistances(mVisSphere.ref_vert_pnt);

    // osc setup
    setupOsc();

    // GUI setup
    // Set Height in ofxDatGuiTheme.h Line 174
    mGui = new ofxDatGui(ofxDatGuiAnchor::TOP_LEFT);
    mGui->addSlider(mGuiFaderNames[0], 0.001, 50.0, 1.0)->setPrecision(7);    
    mGui->addSlider(mGuiFaderNames[1], 0.001, 25.0, 1.0)->setPrecision(7);
    mGui->addSlider(mGuiFaderNames[2], 0.001, 12.0, 0.5)->setPrecision(7);
    mGui->addSlider(mGuiFaderNames[3], 0.001, 12.0, 1.0)->setPrecision(7);
    mGui->addSlider(mGuiFaderNames[4], 0.0, 0.9, 0.0)->setPrecision(7);
    mGui->addSlider(mGuiFaderNames[5], 0.0, 0.9, 0.9)->setPrecision(7);
    mGui->addSlider(mGuiFaderNames[6], 1.0, 100.0, 50.0)->setPrecision(7);
    
    mGui->addToggle("RETRIGGER_AT_INDEX");
    mGui->addToggle("RETRIGGER_AT_END");
    mGui->addToggle("CONTINOUS_TRIGGER");
    
    mGui->addToggle("METRONOME_TRIGGER");
    mGui->addToggle("GRAIN_SHUFFLE");
    mGui->addDropdown("SELECT_WINDOW_FUNCTION", windows);
    
    mGui->addToggle("UNLOCK_PRESET");
    mGui->addButton("READ_WRITE_PRST");
    mGui->addTextInput("SEL_PRST_NR");

    mGui->onSliderEvent(this, &ofApp::onSliderEvent);
    mGui->onDropdownEvent(this, &ofApp::onDropdownEvent);
    mGui->onToggleEvent(this, &ofApp::onToggleEvent);
    mGui->onButtonEvent(this, &ofApp::onButtonEvent);
    mGui->onTextInputEvent(this, &ofApp::onTextInputEvent);
}

//--------------------------------------------------------------
void ofApp::selectAudioDevice()
{
    mSettings.setApi(ofSoundDevice::Api::OSX_CORE);
    mDevices = mSoundStream.getDeviceList(ofSoundDevice::Api::OSX_CORE);

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
    mSettings.numInputChannels = 0;
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
        // Motion Capturing Input 23456
        // Sensor Input 9001
        mOscReceiver = new dab::OscReceiver("MocapReceiver", 9004);
        mOscReceiver->registerOscListener(std::weak_ptr<ofApp>(mSelf));
        mOscReceiver->start();

        mOscSendAddress = "127.0.0.1";
        mOscSendPort = 9003;
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
        
        std::cout << "address " << address << "\n";

        const std::vector<dab::_OscArg*>& arguments = pMessage->arguments();

        if (IMU_SENSOR)
        {
            if (address == "/imu/2")
            {
                mArduinoQuat[0] = *arguments[0];
                mArduinoQuat[1] = *arguments[1];
                mArduinoQuat[2] = *arguments[2];
                mArduinoQuat[3] = *arguments[3];
            }
        }
        else
        {
            if (address == "/mocap/joint/pos_world" || address == "/mocap/0/joint/pos_world")
            {
                size_t jointCount = mJointPositions.size();
                size_t argCount = arguments.size();

                jointCount = std::min(jointCount, argCount / 3);

                for (int jI = 0, aI = 0; jI < jointCount; ++jI, aI += 3)
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
void ofApp::updateOscSender()
{
    try
    {
        {
            if (IMU_SENSOR)
            {
                std::string messageAddress = "/imu/2";
                std::shared_ptr<dab::OscMessage> message(new dab::OscMessage(messageAddress));

                message->add(mArduinoQuat[0]);
                message->add(mArduinoQuat[1]);
                message->add(mArduinoQuat[2]);
                message->add(mArduinoQuat[3]);

                mOscSender->send(message);
            }
            else
            {
                std::string messageAddress = "/mocap/0/joint/pos_world";
                std::shared_ptr<dab::OscMessage> message(new dab::OscMessage(messageAddress));

                for (int pI = 0; pI < mJointPositions.size(); ++pI)
                {
                    message->add(mJointPositions[pI].x);
                    message->add(mJointPositions[pI].y);
                    message->add(mJointPositions[pI].z);
                }
                mOscSender->send(message);
            }
        }
    }
    catch (dab::Exception& e)
    {
        std::cout << e << "\n";
    }
}

//--------------------------------------------------------------
void ofApp::detectSensorCollisions(const glm::vec3& pSensorPosition)
{
    mCollisionIndices.clear();

    int refPointCount = mVisSphere.ref_vert_pnt.size();
    float minColDistance = MIN_COL_DISTANCE;

    for (int iPI = 0; iPI < refPointCount; ++iPI)
    {
        const glm::vec3& instrPoint = mVisSphere.ref_vert_pnt[iPI];

        float dist = glm::length(pSensorPosition - instrPoint);

        if (dist <= minColDistance)
        {
            mCollisionIndices.push_back(iPI);
        }
    }
}

//--------------------------------------------------------------
void ofApp::detectJointCollisions(const std::vector<glm::vec3>& pJointPositions)
{
    mCollisionIndices.clear();

    int refPointCount = mVisSphere.ref_vert_pnt.size();
    float minColDistance = MIN_COL_DISTANCE;

    for (int iPI = 0; iPI < refPointCount; ++iPI)
    {
        const glm::vec3& instrPoint = mVisSphere.ref_vert_pnt[iPI];

        for (auto jointPoint : pJointPositions)
        {
            float dist = glm::length(jointPoint - instrPoint);

            if (dist <= minColDistance)
            {
                mCollisionIndices.push_back(iPI);
            }
        }
    }
}

//--------------------------------------------------------------
void ofApp::triggerAudio(std::vector<std::size_t> pCollisionIndices)
{
    float playbackSpeed = ofRandom(PITCH_RAND_MIN, PITCH_RAND_MAX);

    if (RETRIGGER_AT_INDEX)
    {
        std::sort(pCollisionIndices.begin(), pCollisionIndices.end());
        std::sort(mCollisionIndicesBefore.begin(), mCollisionIndicesBefore.end());

        std::vector<std::size_t> intersection;
        std::set_intersection(std::begin(pCollisionIndices), std::end(pCollisionIndices), std::begin(mCollisionIndicesBefore), std::end(mCollisionIndicesBefore), std::back_inserter(intersection));

        std::vector colChanges = set_difference(mCollisionIndicesBefore, intersection);
        
        for (std::size_t colChangeIndex : colChanges)
        {
            if (RETRIGGER_AT_END)
            {
                if (mAudioPlayers[colChangeIndex].isPlaying() == false)
                {
                    mAudioPlayers[colChangeIndex].start(playbackSpeed, SAMPLE_POS_START, SAMPLE_POS_END);
                }
            }

            if (CONTINOUS_TRIGGER)
            {
                mAudioPlayers[colChangeIndex].start(playbackSpeed, SAMPLE_POS_START, SAMPLE_POS_END);
            }
        }

        mCollisionIndicesBefore.resize(pCollisionIndices.size());
        std::copy_n(pCollisionIndices.begin(), pCollisionIndices.size(), mCollisionIndicesBefore.begin());
    }

    if (!RETRIGGER_AT_INDEX)
    {
        for (std::size_t colIndex : pCollisionIndices)
        {
            if (RETRIGGER_AT_END)
            {
                if (mAudioPlayers[colIndex].isPlaying() == false)
                {
                    mAudioPlayers[colIndex].start(playbackSpeed, SAMPLE_POS_START, SAMPLE_POS_END);
                }
            }

            if (CONTINOUS_TRIGGER)
            {
                mAudioPlayers[colIndex].start(playbackSpeed, SAMPLE_POS_START, SAMPLE_POS_END);
            }
        }
    }

    if (METRONOME_TRIGGER)
    {
        if (mAudioPlayerMetro.play((METRO_FREQUENCY * mAudioBufferSize) + (ofRandom(0, 1) * 2 - 1) * (METRO_RAND_FREQ * mAudioBufferSize)) == true)
        {
            for (std::size_t colIndex : pCollisionIndices)
            {
                sfPlayer tempAudioPlayer = mAudioPlayers[colIndex];
                mAudioGrains.push_back(tempAudioPlayer);
                mAudioGrains[mAudioGrains.size() - 1].start(playbackSpeed, SAMPLE_POS_START, SAMPLE_POS_END);

                if (GRAIN_SHUFFLE == true)
                {
                    auto range = std::default_random_engine{};
                    std::shuffle(std::begin(mAudioGrains), std::end(mAudioGrains), range);
                }

                for (size_t grainCount = 0; grainCount < mAudioGrains.size(); grainCount++)
                {
                    if (mAudioGrains[grainCount].isPlaying() == false)
                    {
                        mAudioGrains.erase(mAudioGrains.begin() + grainCount);
                    }
                }
            }
        }
    }
}

//--------------------------------------------------------------
void ofApp::drawAudioTrigger(std::vector<std::size_t> pCollisionIndices)
{
    for (auto colIndex = pCollisionIndices.rbegin(); colIndex != pCollisionIndices.rend(); ++colIndex)
    {
        mVisSphere.drawExitingPosition(*colIndex, ofColor(255));
    }
}

//--------------------------------------------------------------
void ofApp::calculate2dDistances(const std::vector<glm::vec3>& pSphereVertRefPoints)
{
    for (size_t refPointCount = 1; refPointCount < pSphereVertRefPoints.size() - 1; refPointCount += 10)
    {
        mCollisionRefVertDist.push_back(mConv3dCoord.distance(pSphereVertRefPoints[refPointCount], pSphereVertRefPoints[refPointCount + 5]) / RADIUS);
    }

    for (size_t distCount = 0; distCount < mCollisionRefVertDist.size(); distCount++)
    {
        std::cout << "distances " << mCollisionRefVertDist[distCount] << std::endl;
    }
}

//--------------------------------------------------------------
void ofApp::update()
{
    std::size_t checkForActivePlayers = 0;

    for (size_t sfPlayerIndex = 0; sfPlayerIndex < NUMBER_OF_SOUNDFILES; sfPlayerIndex++)
    {
        if (mAudioPlayers[sfPlayerIndex].isPlaying() == true)
        {
            checkForActivePlayers += 1;
        }
    }
    // std::cout << checkForActivePlayers << std::endl;
}

//--------------------------------------------------------------
void ofApp::draw()
{
    ofEnableDepthTest();

    mVisCam.begin();

    ofPushMatrix();

    ofTranslate(ofGetWidth() * 0.5, ofGetHeight() * 0.5, 0);

    if (drawVerticies) mVisSphere.drawVerticies(ofColor(255, 0, 0));
    if (drawMesh) mVisSphere.drawMesh(ofColor(0));
    if (drawIndicies) mVisSphere.drawIndicies(ofColor(255));

    mVisSphere.drawExitingPosition(colIndexTestCount, ofColor(255));

    if (mAudioThreadSync.load(memory_order_acquire))
    {
        mThreadedCollisionIndices = mCollisionIndices;
        mAudioThreadSync.store(false, memory_order_release);
    }
    drawAudioTrigger(mThreadedCollisionIndices);

    if (IMU_SENSOR)
    {
        ofDrawSphere(x * mVisSphere.radius, y * mVisSphere.radius, z * mVisSphere.radius, 10);
    }
    else
    {
        ofSetColor(ofColor(0, 255, 0));
        for (auto jointPos : mJointPositions)
        {
            ofDrawSphere(jointPos, 1.0);
        }
    }

    ofPopMatrix();

    ofDisableDepthTest();

    mVisCam.end();

    updateOscSender();
}

//--------------------------------------------------------------
void ofApp::audioOut(ofSoundBuffer& outBuffer)
{
    updateOsc();

    if (IMU_SENSOR)
    {
        float angle_rad = glm::acos(mArduinoQuat[3]) * 2;
        float angle_deg = angle_rad * 180 / PI;
        x = mArduinoQuat[0] / glm::sin(angle_rad / 2);
        y = mArduinoQuat[1] / glm::sin(angle_rad / 2);
        z = mArduinoQuat[2] / glm::sin(angle_rad / 2);

        detectSensorCollisions(glm::vec3(x * mVisSphere.radius, y * mVisSphere.radius, z * mVisSphere.radius));
    }
    else
    {
        detectJointCollisions(mJointPositions);
    }

    if (mAudioTrigger) triggerAudio(mCollisionIndices);

    size_t sum = 0;
    for (size_t colIndices = 0; colIndices < mCollisionIndices.size(); colIndices++)
    {     
        sum += colIndices;
        mAudioPosMean = sum / mCollisionIndices.size();
        mAudioPosInDegree = (colIndices + 1) * (360 / COLS);
        mAudioDistanceIndex = ((std::size_t)(mAudioPosMean / mVisSphere.ref_vert_pnt.size())) * mCollisionRefVertDist.size();
    }

    for (size_t buffer_count = 0; buffer_count < outBuffer.getNumFrames(); buffer_count++)
    {
        if (METRONOME_TRIGGER)
        {
            for (size_t grainCount = 0; grainCount < mAudioGrains.size(); grainCount++)
            {
                mAudioOutput += mAudioGrains[grainCount].play(mGuiSelectWindow) * 0.5;
            }
        }
        else
        {
            for (size_t player_count = 0; player_count < NUMBER_OF_SOUNDFILES; player_count++)
            {
                mAudioOutput += mAudioPlayers[player_count].play(mGuiSelectWindow) * 0.5;
            }
        }

        mAudioOutput = mAudioOutput * 0.5;

        if (mAudioNumOutputChannels == 2)
        {
            outBuffer[buffer_count * outBuffer.getNumChannels() + 0] = mAudioAmbiDecode[0].play(mAudioAmbiEncode[0].play(mAudioOutput, mAudioPosInDegree, mCollisionRefVertDist[mAudioDistanceIndex]), 45);
            outBuffer[buffer_count * outBuffer.getNumChannels() + 1] = mAudioAmbiDecode[1].play(mAudioAmbiEncode[1].play(mAudioOutput, mAudioPosInDegree, mCollisionRefVertDist[mAudioDistanceIndex]), 270);
        }

        if (mAudioNumOutputChannels == 4)
        {
            outBuffer[buffer_count * outBuffer.getNumChannels() + 0] = mAudioAmbiDecode[0].play(mAudioAmbiEncode[0].play(mAudioOutput, mAudioPosInDegree, mCollisionRefVertDist[mAudioDistanceIndex]), 45);
            outBuffer[buffer_count * outBuffer.getNumChannels() + 1] = mAudioAmbiDecode[1].play(mAudioAmbiEncode[1].play(mAudioOutput, mAudioPosInDegree, mCollisionRefVertDist[mAudioDistanceIndex]), 135);
            outBuffer[buffer_count * outBuffer.getNumChannels() + 2] = mAudioAmbiDecode[2].play(mAudioAmbiEncode[2].play(mAudioOutput, mAudioPosInDegree, mCollisionRefVertDist[mAudioDistanceIndex]), 225);
            outBuffer[buffer_count * outBuffer.getNumChannels() + 3] = mAudioAmbiDecode[3].play(mAudioAmbiEncode[3].play(mAudioOutput, mAudioPosInDegree, mCollisionRefVertDist[mAudioDistanceIndex]), 270);
        }

        if (mAudioNumOutputChannels == 6)
        {
            outBuffer[buffer_count * outBuffer.getNumChannels() + 0] = mAudioAmbiDecode[0].play(mAudioAmbiEncode[0].play(mAudioOutput, mAudioPosInDegree, mCollisionRefVertDist[mAudioDistanceIndex]), 60);
            outBuffer[buffer_count * outBuffer.getNumChannels() + 1] = mAudioAmbiDecode[1].play(mAudioAmbiEncode[1].play(mAudioOutput, mAudioPosInDegree, mCollisionRefVertDist[mAudioDistanceIndex]), 120);
            outBuffer[buffer_count * outBuffer.getNumChannels() + 2] = mAudioAmbiDecode[2].play(mAudioAmbiEncode[2].play(mAudioOutput, mAudioPosInDegree, mCollisionRefVertDist[mAudioDistanceIndex]), 180);
            outBuffer[buffer_count * outBuffer.getNumChannels() + 3] = mAudioAmbiDecode[3].play(mAudioAmbiEncode[3].play(mAudioOutput, mAudioPosInDegree, mCollisionRefVertDist[mAudioDistanceIndex]), 240);
            outBuffer[buffer_count * outBuffer.getNumChannels() + 4] = mAudioAmbiDecode[4].play(mAudioAmbiEncode[4].play(mAudioOutput, mAudioPosInDegree, mCollisionRefVertDist[mAudioDistanceIndex]), 300);
            outBuffer[buffer_count * outBuffer.getNumChannels() + 5] = mAudioAmbiDecode[5].play(mAudioAmbiEncode[5].play(mAudioOutput, mAudioPosInDegree, mCollisionRefVertDist[mAudioDistanceIndex]), 360);
        }

        if (mAudioNumOutputChannels == 8)
        {
            outBuffer[buffer_count * outBuffer.getNumChannels() + 0] = mAudioAmbiDecode[0].play(mAudioAmbiEncode[0].play(mAudioOutput, mAudioPosInDegree, mCollisionRefVertDist[mAudioDistanceIndex]), 45);
            outBuffer[buffer_count * outBuffer.getNumChannels() + 1] = mAudioAmbiDecode[1].play(mAudioAmbiEncode[1].play(mAudioOutput, mAudioPosInDegree, mCollisionRefVertDist[mAudioDistanceIndex]), 90);
            outBuffer[buffer_count * outBuffer.getNumChannels() + 2] = mAudioAmbiDecode[2].play(mAudioAmbiEncode[2].play(mAudioOutput, mAudioPosInDegree, mCollisionRefVertDist[mAudioDistanceIndex]), 135);
            outBuffer[buffer_count * outBuffer.getNumChannels() + 3] = mAudioAmbiDecode[3].play(mAudioAmbiEncode[3].play(mAudioOutput, mAudioPosInDegree, mCollisionRefVertDist[mAudioDistanceIndex]), 180);
            outBuffer[buffer_count * outBuffer.getNumChannels() + 4] = mAudioAmbiDecode[4].play(mAudioAmbiEncode[4].play(mAudioOutput, mAudioPosInDegree, mCollisionRefVertDist[mAudioDistanceIndex]), 225);
            outBuffer[buffer_count * outBuffer.getNumChannels() + 5] = mAudioAmbiDecode[5].play(mAudioAmbiEncode[5].play(mAudioOutput, mAudioPosInDegree, mCollisionRefVertDist[mAudioDistanceIndex]), 270);
            outBuffer[buffer_count * outBuffer.getNumChannels() + 6] = mAudioAmbiDecode[6].play(mAudioAmbiEncode[6].play(mAudioOutput, mAudioPosInDegree, mCollisionRefVertDist[mAudioDistanceIndex]), 315);
            outBuffer[buffer_count * outBuffer.getNumChannels() + 7] = mAudioAmbiDecode[7].play(mAudioAmbiEncode[7].play(mAudioOutput, mAudioPosInDegree, mCollisionRefVertDist[mAudioDistanceIndex]), 360);
        }

        writeFileToBuffer(outBuffer[buffer_count]);
        mAudioOutput = 0;
    }

    mAudioThreadSync.store(true, memory_order_release);
}

//--------------------------------------------------------------
void ofApp::writeFileSetup(const std::string& pSfName)
{
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
void ofApp::keyPressed(int key)
{
    // Sphere
    if (key == 'y') drawVerticies = !drawVerticies;
    if (key == 'x') drawMesh = !drawMesh;
    if (key == 'c') drawIndicies = !drawIndicies;
    if (key == 'v') drawExitingPos = !drawExitingPos;
    // audio parameter
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

    if (key == 'r') writeFileSetup("audioTest.wav");;
    if (key == 'w') writeFileToDisk(writeFileSampleBuffer);
    // test
    if (key == ' ')
    {
        colIndexTestCount = (colIndexTestCount += 1) % mVisSphere.ref_vert_pnt.size();
        mVisSphere.drawExitingPosition(colIndexTestCount, ofColor(255));
        mAudioPlayers[colIndexTestCount].start(ofRandom(PITCH_RAND_MIN, PITCH_RAND_MAX), SAMPLE_POS_START, SAMPLE_POS_END);
    }
}

//--------------------------------------------------------------
void ofApp::onSliderEvent(ofxDatGuiSliderEvent e)
{
    if (e.target->is(mGuiFaderNames[0])) METRO_FREQUENCY = e.value;
    if (e.target->is(mGuiFaderNames[1])) METRO_RAND_FREQ = e.value;
    if (e.target->is(mGuiFaderNames[2])) PITCH_RAND_MIN = e.value;
    if (e.target->is(mGuiFaderNames[3])) PITCH_RAND_MAX = e.value;
    if (e.target->is(mGuiFaderNames[4])) SAMPLE_POS_START = e.value;
    if (e.target->is(mGuiFaderNames[5])) SAMPLE_POS_END = e.value;
    if (e.target->is(mGuiFaderNames[6])) MIN_COL_DISTANCE = e.value;
}

//--------------------------------------------------------------
void ofApp::onDropdownEvent(ofxDatGuiDropdownEvent e)
{
    if (e.target->is("SELECT_WINDOW_FUNCTION")) mGuiSelectWindow = e.child; std::cout << mGuiSelectWindow << endl;
}

//--------------------------------------------------------------
void ofApp::onToggleEvent(ofxDatGuiToggleEvent e)
{
    if (e.target->is("RETRIGGER_AT_INDEX"))
    {
        RETRIGGER_AT_INDEX = e.checked;
        if (RETRIGGER_AT_INDEX)
        {
            mGui->getToggle("METRONOME_TRIGGER")->setChecked(false);
            METRONOME_TRIGGER = false;
        }
    }
    if (e.target->is("RETRIGGER_AT_END"))
    {
        RETRIGGER_AT_END = e.checked;
        if (RETRIGGER_AT_END)
        {
            mGui->getToggle("CONTINOUS_TRIGGER")->setChecked(false);
            CONTINOUS_TRIGGER = false;
        }
    }
    if (e.target->is("CONTINOUS_TRIGGER"))
    {
        CONTINOUS_TRIGGER = e.checked;
        if (CONTINOUS_TRIGGER)
        {
            mGui->getToggle("RETRIGGER_AT_END")->setChecked(false);
            RETRIGGER_AT_END = false;
        }
    }
    if (e.target->is("METRONOME_TRIGGER"))
    {
        METRONOME_TRIGGER = e.checked;
        if (METRONOME_TRIGGER)
        {
            mGui->getToggle("RETRIGGER_AT_INDEX")->setChecked(false);
            RETRIGGER_AT_INDEX = false;
        }
    }
    if (e.target->is("GRAIN_SHUFFLE")) GRAIN_SHUFFLE = e.checked;
    if (e.target->is("UNLOCK_PRESET")) mGuiUnlockPresets = e.checked;
}

//--------------------------------------------------------------
void ofApp::onTextInputEvent(ofxDatGuiTextInputEvent e)
{
    if (e.target->is("SEL_PRST_NR")) mGuiSelectPresetNumber = e.text;
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
void ofApp::saveGuiValuesAsFile(const std::string& pGuiSelectPresetNumber)
{
    std::ostringstream guiValues;

    for (int i = 0; i < mGuiFaderNames.size(); ++i)
    {
        std::cout << mGuiFaderNames[i] << " " << mGui->getSlider(mGuiFaderNames[i])->getValue() << " " << std::endl;
        guiValues << mGuiFaderNames[i] << " " << mGui->getSlider(mGuiFaderNames[i])->getValue() << " " << std::endl;
    }

    std::string guiWriteFileName = "preset_number_" + pGuiSelectPresetNumber;
    mGuiFileIO.write(guiValues.str(), ofToDataPath(guiWriteFileName));
}

//--------------------------------------------------------------
void ofApp::readGuiValuesFromFile(const std::string& pGuiSelectPresetNumber)
{
    std::string fileValues = {};

    std::string guiReadFileName = "preset_number_" + pGuiSelectPresetNumber;
    mGuiFileIO.read(ofToDataPath(guiReadFileName), fileValues);

    const char delimiter = ' ';
    std::vector<std::string> outputArray;
    std::istringstream streamData(fileValues);
    std::string token;

    while (std::getline(streamData, token, delimiter))
    {
        token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
        outputArray.push_back(token);
    }

    for (int i = 0; i < outputArray.size() - 1; i += 2)
    {
        mGui->getSlider(outputArray[i])->setValue(std::stof(outputArray[i + 1]));
    }
}

//--------------------------------------------------------------
void ofApp::exit()
{
    delete[] mAudioPlayers;
    delete[] mAudioAmbiEncode;
    delete[] mAudioAmbiDecode;
}

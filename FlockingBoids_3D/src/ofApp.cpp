#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup()
{
    float radius = 300;

    sphere.setRadius(radius);
    sphere.setResolution(10);

    attractorPoints = sphere.getMesh().getVertices();

    for (int i = 0; i < 2000; i++) 
    {
        float theta = ofRandom(TWO_PI);
        float phi = ofRandom(PI);
        float r = radius + ofRandom(-10, 10);
        float x = r * sin(phi) * cos(theta);
        float y = r * sin(phi) * sin(theta);
        float z = r * cos(phi);

        flock.addBoid(Boid(x, y, z));
    }

    ofBackground(255);

    pointLight.setup();
    pointLight.setPosition(200, 200, 600);  
    pointLight.setDiffuseColor(ofFloatColor(1.0, 1.0, 1.0));   
    pointLight.setSpecularColor(ofFloatColor(1.0, 1.0, 1.0));  

    ofEnableLighting();
    pointLight.enable();

    ofSetVerticalSync(true);

    flocking_GUI = new ofxDatGui(ofxDatGuiAnchor::TOP_LEFT);
    flocking_GUI->addSlider(flocking_GUI_FaderNames[0], 0.0, 200.0, 50.0)->setPrecision(7);
    flocking_GUI->addSlider(flocking_GUI_FaderNames[1], 0.0, 5.0, 2.0)->setPrecision(7);
    flocking_GUI->addSlider(flocking_GUI_FaderNames[2], 0.0, 200.0, 80.0)->setPrecision(7);
    flocking_GUI->addSlider(flocking_GUI_FaderNames[3], 0.0, 5.0, 1.0)->setPrecision(7);
    flocking_GUI->addSlider(flocking_GUI_FaderNames[4], 0.0, 200.0, 200.0)->setPrecision(7);
    flocking_GUI->addSlider(flocking_GUI_FaderNames[5], 0.0, 5.0, 1.5)->setPrecision(7);
    flocking_GUI->addSlider(flocking_GUI_FaderNames[6], 0.1, 10.0, 3.5)->setPrecision(7);
    flocking_GUI->addSlider(flocking_GUI_FaderNames[7], 0.1, 1.0, 0.2)->setPrecision(7);
    flocking_GUI->addSlider(flocking_GUI_FaderNames[8], 0.0, 1.0, 0.1)->setPrecision(7);
    flocking_GUI->addSlider(flocking_GUI_FaderNames[9], 0.0, 1.0, 0.12)->setPrecision(7);
    flocking_GUI->addSlider(flocking_GUI_FaderNames[10], 0.0, 320.0, 350.0)->setPrecision(7);
    flocking_GUI->addSlider(flocking_GUI_FaderNames[11], 0.0, 180.0, 90.0)->setPrecision(7);
    flocking_GUI->addSlider(flocking_GUI_FaderNames[12], 0.0, 360.0, 0.0)->setPrecision(7);

    flocking_GUI->onSliderEvent(this, &ofApp::onSliderEvent);

    presetControl_GUI = new ofxDatGui(ofxDatGuiAnchor::BOTTOM_LEFT);
    presetControl_GUI->addToggle("UNLOCK_PRESET");
    presetControl_GUI->addButton("READ_WRITE_PRST");
    presetControl_GUI->addTextInput("SEL_PRST_NR")->setInputType(ofxDatGuiInputType::NUMERIC);

    presetControl_GUI->onToggleEvent(this, &ofApp::onToggleEvent);
    presetControl_GUI->onButtonEvent(this, &ofApp::onButtonEvent);
    presetControl_GUI->onTextInputEvent(this, &ofApp::onTextInputEvent);

    boidFbo.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA);
    boidFbo.begin();
    ofClear(255, 255, 255, 255);
    boidFbo.end();

    sender.setup(host, port);

    cam.removeAllInteractions();
    cam.addInteraction(ofEasyCam::TRANSFORM_ROTATE, OF_MOUSE_BUTTON_RIGHT);
    cam.setLensOffset(ofVec2f(-0.3, 0));
}

//--------------------------------------------------------------
void ofApp::update()
{
    float thetaRad = glm::radians(theta);
    float phiRad = glm::radians(phi);
    movingTarget.x = radius * sin(thetaRad) * cos(phiRad);
    movingTarget.y = radius * sin(thetaRad) * sin(phiRad);
    movingTarget.z = radius * cos(thetaRad);

    flock.run(&attractorPoints, 
        &movingTarget, 
        separationDistance, 
        separationWeight, 
        alignmentDistance, 
        alignmentWeight, 
        cohesionDistance, 
        cohesionWeight, 
        maxSpeed, 
        maxForce, 
        targetSeekStrength, 
        sphereSeekStrength);

    const std::vector<Boid>& boids = flock.getBoids();
    ofxOscMessage m;
    m.setAddress("/boid/positions");

    int limit = std::min((int)boids.size(), 250);
    for (int i = 0; i < limit; ++i) {
        glm::vec3 pos = boids[i].getPosition();
        m.addFloatArg(pos.x);
        m.addFloatArg(pos.y);
        m.addFloatArg(pos.z);
    }
    sender.sendMessage(m, false);
 }


//--------------------------------------------------------------
void ofApp::draw()
{
    boidFbo.begin();
    ofClear(255, 255, 255, 255);

    ofEnableDepthTest();
    ofEnableLighting();

    cam.begin();   
    
    flock.draw();

    if (drawDebugElements)
    {
        glPointSize(5);
        sphere.drawVertices();

        ofSetColor(255, 0, 0);
        ofDrawSphere(movingTarget, 5);
    }

    cam.end();

    ofDisableLighting();
    ofDisableDepthTest();

    boidFbo.end();

    ofSetColor(255);
    boidFbo.draw(0, 0);
}

//--------------------------------------------------------------
void ofApp::onSliderEvent(ofxDatGuiSliderEvent e)
{
    if (e.target->is(flocking_GUI_FaderNames[0])) separationDistance = e.value;
    if (e.target->is(flocking_GUI_FaderNames[1])) separationWeight = e.value;
    if (e.target->is(flocking_GUI_FaderNames[2])) alignmentDistance = e.value;
    if (e.target->is(flocking_GUI_FaderNames[3])) alignmentWeight = e.value;
    if (e.target->is(flocking_GUI_FaderNames[4])) cohesionDistance = e.value;
    if (e.target->is(flocking_GUI_FaderNames[5])) cohesionWeight = e.value;
    if (e.target->is(flocking_GUI_FaderNames[6])) maxSpeed = e.value;
    if (e.target->is(flocking_GUI_FaderNames[7])) maxForce = e.value;
    if (e.target->is(flocking_GUI_FaderNames[8])) targetSeekStrength = e.value;
    if (e.target->is(flocking_GUI_FaderNames[9])) sphereSeekStrength = e.value;
    if (e.target->is(flocking_GUI_FaderNames[10])) radius = e.value;
    if (e.target->is(flocking_GUI_FaderNames[11])) theta = e.value;
    if (e.target->is(flocking_GUI_FaderNames[12])) phi = e.value;
}

//--------------------------------------------------------------
void ofApp::onToggleEvent(ofxDatGuiToggleEvent e)
{
    if (e.target->is("UNLOCK_PRESET")) mGuiUnlockPresets = e.checked;
}

//--------------------------------------------------------------
void ofApp::onTextInputEvent(ofxDatGuiTextInputEvent e)
{
    if (e.target->is("SEL_PRST_NR")) {
        mGuiSelectPresetNumber = e.text;
        return;
    }
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

    for (const auto& name : flocking_GUI_FaderNames)
    {
        json["flocking_GUI"][name] = flocking_GUI->getSlider(name)->getValue();
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

    for (const auto& name : flocking_GUI_FaderNames)
    {
        if (json["flocking_GUI"].contains(name))
        {
            float val = json["flocking_GUI"][name].get<float>();
            flocking_GUI->getSlider(name)->setValue(val);

            if (name == flocking_GUI_FaderNames[0]) separationDistance = val;
            else if (name == flocking_GUI_FaderNames[1]) separationWeight = val;
            else if (name == flocking_GUI_FaderNames[2]) alignmentDistance = val;
            else if (name == flocking_GUI_FaderNames[3]) alignmentWeight = val;
            else if (name == flocking_GUI_FaderNames[4]) cohesionDistance = val;
            else if (name == flocking_GUI_FaderNames[5]) cohesionWeight = val;
            else if (name == flocking_GUI_FaderNames[6]) maxSpeed = val;
            else if (name == flocking_GUI_FaderNames[7]) maxForce = val;
            else if (name == flocking_GUI_FaderNames[8]) targetSeekStrength = val;
            else if (name == flocking_GUI_FaderNames[9]) sphereSeekStrength = val;
            else if (name == flocking_GUI_FaderNames[10]) radius = val;
            else if (name == flocking_GUI_FaderNames[11]) theta = val;
            else if (name == flocking_GUI_FaderNames[12]) phi = val;
        }
    }
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key)
{
    if (key == ' ') drawDebugElements = !drawDebugElements;
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}

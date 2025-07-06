#include "ofApp.h"
#include "dab_index_map.h"
#include "dab_listener.h"
#include "dab_array.h"
#include <array>
#include <vector>

class MyListenerClass : public dab::UpdateListener
{
public:
	MyListenerClass() {};
	~MyListenerClass() {};

	void notifyUpdate(const dab::AbstractValue& pValue)
	{
		std::cout << pValue << "\n";
	}

	void notifyUpdate()
	{
		std::cout << "void" << "\n";
	}

protected:
};

class MyNotifyingClass : public dab::UpdateNotifier
{
public:
	void update()
	{
		int listenerCount = mListeners.size();
		for (int lI = 0; lI<listenerCount; ++lI)
		{
			std::shared_ptr<dab::UpdateListener> listener = mListeners[lI].lock();
			listener->notifyUpdate(dab::Value<MyNotifyingClass*>(this));
		}
	}

protected:
	float myVariable = 5.0;
};

//--------------------------------------------------------------
void ofApp::setup()
{
	// example use of notifier and listener class
	MyNotifyingClass notifier;
	std::shared_ptr<MyListenerClass> listener(std::shared_ptr<MyListenerClass>(new MyListenerClass()));

	notifier.addListener(listener);
	notifier.update();

	// example use of array class
	dab::Array<int> array1( { 1, 3, 5, 7 } );
	dab::Array<int> array2( { 1, 3, 5, 7 } );
	dab::Array<int> array3( { 1, 5, 7 } );
	dab::Array<int> array4( { 1, 3, 6, 7 } );
	    
	std::cout << array1 << "\n";
	    
	std::cout << "a1 == a2 " << (array1 == array2) << "\n";
	std::cout << "a1 == a3 " << (array1 == array3) << "\n";
	std::cout << "a1 == a4 " << (array1 == array4) << "\n";
}

//--------------------------------------------------------------
void ofApp::update() {

}

//--------------------------------------------------------------
void ofApp::draw() {

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key) {

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h) {

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg) {

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {

}

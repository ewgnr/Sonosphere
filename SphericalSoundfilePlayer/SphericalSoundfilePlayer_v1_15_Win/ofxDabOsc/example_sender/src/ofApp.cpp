#include "ofApp.h"
#include <sstream>
#include "osc/OscReceivedElements.h"
#include "osc/OscOutboundPacketStream.h"
#include "dab_osc_arg.h"
#include "dab_osc_message.h"
#include "dab_osc_sender.h"

template<class T>
void print(T var, std::string end = "\n") {
	std::cout << var << " is of type " << typeid(var).name() << end;
}

//--------------------------------------------------------------
void ofApp::setup()
{
	try
	{
		dab::OscUtils& oscUtils = dab::OscUtils::get();

		oscSender = new dab::OscSender("OscSender", "127.0.0.1", 9001);
		oscSender->start();

		//        std::shared_ptr<dab::OscMessage> message( new dab::OscMessage("/HelloWorld") );
		//
		//        int charCount = 50;
		//        int intCount = 50;
		//        int longCount = 50;
		//        int floatCount = 50;
		//        int doubleCount = 50;
		//        int stringSize = 2000;
		//        int charArraySize = 2000;
		//        int intArraySize = 2000;
		//        int longArraySize = 2000;
		//        int floatArraySize = 2000;
		//        int doubleArraySize = 2000;
		//
		//        for(int i=0; i<charCount; ++i)
		//        {
		//            message->add( (char)( i % 26 + 97 ) );
		//        }
		//
		//        for(int i=0; i<intCount; ++i)
		//        {
		//            message->add( (int)i );
		//        }
		//
		//        for(int i=0; i<longCount; ++i)
		//        {
		//            message->add( (long)i );
		//        }
		//
		//        for(int i=0; i<floatCount; ++i)
		//        {
		//            message->add( (float)i );
		//        }
		//
		//        for(int i=0; i<doubleCount; ++i)
		//        {
		//            message->add( (double)i );
		//        }
		//
		//        std::string string;
		//        for(int i=0; i<stringSize; ++i)
		//        {
		//            string += (char)( i % 26 + 97 );
		//        }
		//        message->add( string );
		//
		//        char charArray[charArraySize];
		//        for(int i=0; i<charArraySize; ++i)
		//        {
		//            charArray[i] = (char)( i % 26 + 97 );
		//        }
		//        message->add( charArray, charArraySize );
		//
		//        int intArray[intArraySize];
		//        for(int i=0; i<intArraySize; ++i)
		//        {
		//            intArray[i] = (int)i;
		//        }
		//        message->add( intArray, intArraySize );
		//
		//        long longArray[longArraySize];
		//        for(int i=0; i < longArraySize; ++i)
		//        {
		//            longArray[i] = (long)i;
		//        }
		//        message->add( longArray, longArraySize );
		//
		//        float floatArray[floatArraySize];
		//        for(int i=0; i < floatArraySize; ++i)
		//        {
		//            floatArray[i] = (float)i;
		//        }
		//        message->add( floatArray, floatArraySize );
		//
		//        double doubleArray[doubleArraySize];
		//        for(int i=0; i < doubleArraySize; ++i)
		//        {
		//            doubleArray[i] = (double)i;
		//        }
		//        message->add( doubleArray, doubleArraySize );
		//
		////        std::cout << "full message\n";
		////        std::cout << *message << "\n";
		//
		//        oscSender->queue(message);

	}
	catch (dab::Exception& e)
	{
		std::cout << e << "\n";
	}

}

//--------------------------------------------------------------
void ofApp::update()
{
	std::string messageAddress = "/hello " + to_string(mSentMessageCount);
	std::shared_ptr<dab::OscMessage> message(new dab::OscMessage(messageAddress));
	//std::string testString = "testString";

	//int64_t time = ofGetElapsedTimeMillis();
	//message->add(time);

	message->add(0.5f);
	message->add(7);
	//message->add(testString);

	int someManyInts[100];
	for (int i = 0; i<100; ++i) someManyInts[i] = i;

	message->add(someManyInts, 100);

	oscSender->send(message);

	//    usleep(100);
	//
	//    mSentMessageCount++;

	//std::cout << "mSentMessageCount " << mSentMessageCount << "\n";

	//    std::string messageAddress = "/Message Nr " + to_string(mSentMessageCount);
	//
	//    std::shared_ptr<dab::OscMessage> message( new dab::OscMessage(messageAddress) );
	//
	//    int charCount = 50;
	//    int intCount = 50;
	//    int longCount = 50;
	//    int floatCount = 50;
	//    int doubleCount = 50;
	//    int stringSize = 2000;
	//    int charArraySize = 2000;
	//    int intArraySize = 2000;
	//    int longArraySize = 2000;
	//    int floatArraySize = 2000;
	//    int doubleArraySize = 2000;
	//
	//    for(int i=0; i<charCount; ++i)
	//    {
	//        message->add( (char)( i % 26 + 97 ) );
	//    }
	//
	//    for(int i=0; i<intCount; ++i)
	//    {
	//        message->add( (int)i );
	//    }
	//
	//    for(int i=0; i<longCount; ++i)
	//    {
	//        message->add( (long)i );
	//    }
	//
	//    for(int i=0; i<floatCount; ++i)
	//    {
	//        message->add( (float)i );
	//    }
	//
	//    for(int i=0; i<doubleCount; ++i)
	//    {
	//        message->add( (double)i );
	//    }
	//
	//    std::string string;
	//    for(int i=0; i<stringSize; ++i)
	//    {
	//        string += (char)( i % 26 + 97 );
	//    }
	//    message->add( string );
	//
	//    char charArray[charArraySize];
	//    for(int i=0; i<charArraySize; ++i)
	//    {
	//        charArray[i] = (char)( i % 26 + 97 );
	//    }
	//    message->add( charArray, charArraySize );
	//
	//    int intArray[intArraySize];
	//    for(int i=0; i<intArraySize; ++i)
	//    {
	//        intArray[i] = (int)i;
	//    }
	//    message->add( intArray, intArraySize );
	//
	//    long longArray[longArraySize];
	//    for(int i=0; i < longArraySize; ++i)
	//    {
	//        longArray[i] = (long)i;
	//    }
	//    message->add( longArray, longArraySize );
	//
	//    float floatArray[floatArraySize];
	//    for(int i=0; i < floatArraySize; ++i)
	//    {
	//        floatArray[i] = (float)i;
	//    }
	//    message->add( floatArray, floatArraySize );
	//
	//    double doubleArray[doubleArraySize];
	//    for(int i=0; i < doubleArraySize; ++i)
	//    {
	//        doubleArray[i] = (double)i;
	//    }
	//    message->add( doubleArray, doubleArraySize );
	//
	//    //        std::cout << "full message\n";
	//    //        std::cout << *message << "\n";
	//
	//    oscSender->queue(message);
	//
	//    usleep(100);
	//
	//    mSentMessageCount++;
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

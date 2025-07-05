/** \file dab_osc_receiver.cpp
 */

#include "dab_osc_receiver.h"
#include "dab_osc_utils.h"

using namespace dab;

OscReceiver::ReceiverThread::ReceiverThread( OscReceiver* pReceiver )
: mReceiver(pReceiver)
{}

void
OscReceiver::ReceiverThread::threadedFunction()
{
	//std::cout << "OscReceiver::ReceiverThread::threadedFunction()\n";
    mReceiver->mReceiveSocket->Run();
}

unsigned int OscReceiver::ListenerThread::sUpdateInterval = 10;

OscReceiver::ListenerThread::ListenerThread( OscReceiver* pReceiver )
: mReceiver(pReceiver)
, mUpdateInterval(sUpdateInterval)
{}

void
OscReceiver::ListenerThread::setUpdateInterval(unsigned int pUpdateInterval)
{
    mUpdateInterval = pUpdateInterval;
}

void
OscReceiver::ListenerThread::threadedFunction()
{
    while(isThreadRunning())
    {
        mReceiver->updateListeners();

		std::this_thread::sleep_for(std::chrono::microseconds(mUpdateInterval));
    }
}


OscReceiver::OscReceiver()
: mName("OscReceiver")
, mReceivingPort( 10000 )
, mMaxMessageCount(-1)
, mReceiveSocket( nullptr )
, mReceiverThread( nullptr )
{}

OscReceiver::OscReceiver( const std::string& pName, unsigned int pPort )
: mName(pName)
, mReceivingPort(pPort)
, mMaxMessageCount(-1)
, mReceiveSocket( nullptr )
, mReceiverThread( nullptr )
{}

OscReceiver::~OscReceiver()
{
    stop();
    
    if( mReceiverThread  != nullptr ) delete mReceiverThread;
    if( mListenerThread  != nullptr ) delete mListenerThread;
    
    if( mReceiveSocket != nullptr )
    {
        mReceiveSocket->Break();
        delete mReceiveSocket;
    }
}

void
OscReceiver::setup(const std::string& pName, unsigned int pPort)
{
    mName = pName;
    
    if( mReceiveSocket != nullptr ) return;
    mReceivingPort = pPort;
}

void
OscReceiver::setListenInterval(unsigned int pListenInterval)
{
    if( mListenerThread != nullptr ) mListenerThread->setUpdateInterval(pListenInterval);
}

void
OscReceiver::setMaxMessageCount(int pMaxMessageCount)
{
    mMaxMessageCount = pMaxMessageCount;
}

void
OscReceiver::registerOscListener( std::weak_ptr<OscListener> pListener )
{
    mMessageListeners.push_back(pListener);
}

const std::string&
OscReceiver::name() const
{
    return mName;
}

unsigned int
OscReceiver::port() const
{
    return mReceivingPort;
}

void
OscReceiver::start()
{
    //std::cout << "OscReceiver::start() begin\n";

    if( mReceiveSocket != nullptr ) return;
    
    mReceiveSocket = new osc::UdpListeningReceiveSocket( osc::IpEndpointName( osc::IpEndpointName::ANY_ADDRESS, mReceivingPort ), this );
    
    mReceiverThread = new ReceiverThread(this);
    mReceiverThread->startThread();
    
    mListenerThread = new ListenerThread(this);
    mListenerThread->startThread();
    
    //std::cout << "OscReceiver::start() end\n";
}

void
OscReceiver::start(unsigned int pListenInterval)
{
    //std::cout << "OscReceiver::start() begin\n";
    
    if( mReceiveSocket != nullptr ) return;
    
    mReceiveSocket = new osc::UdpListeningReceiveSocket( osc::IpEndpointName( osc::IpEndpointName::ANY_ADDRESS, mReceivingPort ), this );
    
    mReceiverThread = new ReceiverThread(this);
    mReceiverThread->startThread();
    
    mListenerThread = new ListenerThread(this);
    mListenerThread->setUpdateInterval(pListenInterval);
    mListenerThread->startThread();
    
    //std::cout << "OscReceiver::start() end\n";
}

void
OscReceiver::stop()
{
    if(mReceiverThread != nullptr && mReceiverThread->isThreadRunning() == true ) mReceiverThread->stopThread();
    if(mListenerThread != nullptr && mListenerThread->isThreadRunning() == true ) mListenerThread->stopThread();
}

void
OscReceiver::ProcessMessage( const osc::ReceivedMessage& pReceivedMessage, const osc::IpEndpointName& pRemoteEndpoint )
{
    mMutex.lock();
    
    try
    {
        //std::cout << "OscReceiver::ProcessMessage begin\n";
		//std::cout << "name " << mName << "\n";
        
        OscUtils& oscUtils = OscUtils::get();

        if( oscUtils.isMultiOscMessage(pReceivedMessage) == false )
        {
            //std::cout << "is single osc message\n";
            
            std::shared_ptr< OscMessage > oscMessage = oscUtils.createMessage(pReceivedMessage);
            
            mReceivedMessages.push_back(oscMessage);
            
            // debug
            //std::cout << "oscMessage\n";
            //std::cout << *oscMessage << "\n";
            // debug done
        }
        else
        {
            //std::cout << "is multi osc message\n";
            
            std::shared_ptr< MultiOscMessage > multiOscMessage;
            unsigned long messageId = oscUtils.messageId(pReceivedMessage);
            
            //std::cout << "messageId " << messageId << "\n";
            
            if( mPendingMultiOscMessages.find(messageId) == mPendingMultiOscMessages.end() ) // new multi osc message
            {
                //std::cout << "new multi message\n";
                //std::cout << "messageId " << messageId << "\n";
                
                multiOscMessage = oscUtils.createMultiOscMessage(pReceivedMessage);
                mPendingMultiOscMessages[messageId] = multiOscMessage;
                
                //std::cout << "array sizes " << multiOscMessage->arraySizes() << "\n";
            }
            else
            {
                //std::cout << "next multi message\n";
                
                multiOscMessage = mPendingMultiOscMessages[messageId];
                oscUtils.updateMultiOscMessage(pReceivedMessage, multiOscMessage);
                
                //std::cout << "array sizes " << multiOscMessage->arraySizes() << "\n";
            }
            
            if( multiOscMessage->complete() == true )
            {
                std::shared_ptr< OscMessage > oscMessage = oscUtils.mergeMessage(multiOscMessage);
                mReceivedMessages.push_back(oscMessage);
                
                mPendingMultiOscMessages.erase(messageId);
                
//                // debug
//                std::cout << "oscMessage\n";
//                std::cout << *oscMessage << "\n";
//                // debug done
            }
        }
        
        //std::cout << "OscReceiver::ProcessMessage end\n";
        
    }
    catch(Exception& e)
    {
        mMutex.unlock();
    
        e += Exception("OSC Error: Failed to process osc::ReceivedMessage", __FILE__, __FUNCTION__, __LINE__);
        
        std::cout << e << "\n";
    }
    
//    osc::ReceivedMessage::const_iterator arg = pMessage.ArgumentsBegin();
//    
//    
//    bool a1 = (arg++)->AsBool();
//    int a2 = (arg++)->AsInt32();
//    float a3 = (arg++)->AsFloat();
//    const char *a4 = (arg++)->AsString();
//    if( arg != pMessage.ArgumentsEnd() )
//        throw osc::ExcessArgumentException();
    
    if(mMaxMessageCount > 0)
    {
        while(mReceivedMessages.size() > mMaxMessageCount) mReceivedMessages.pop_front();
    }
    
    mMutex.unlock();
    
    //updateListeners();
}

void
OscReceiver::flushMessages()
{
    mMutex.lock();
    
    mReceivedMessages.clear();
    
    mMutex.unlock();
}

void
OscReceiver::updateListeners()
{
    //std::cout << "OscReceiver::updateListeners()\n";
	//std::cout << "updateListeners " << mName << "\n";
    
    int availableMessageCount = mReceivedMessages.size();

	//std::cout << "receiver " << mName << " availableMessageCount " << availableMessageCount << "\n";

    if( availableMessageCount == 0 ) return;
    
    std::vector< std::shared_ptr< OscMessage > > receivedMessages;
    
    if( mMutex.try_lock() )
    {
        receivedMessages.resize(availableMessageCount);
        for(int i=0; i<availableMessageCount; ++i)
        {
            receivedMessages[i] = mReceivedMessages[i];
        }

        mReceivedMessages.clear();
        
        mMutex.unlock();
    }
    else return;
    
    //std::cout << "OscReceiver::updateListeners()\n";
    
    int messageListenerCount = mMessageListeners.size();
    
	//std::cout << "receiver " << mName << " messageListenerCount " << messageListenerCount << "\n";
    
    if( messageListenerCount == 0 ) return;
    
    for(int mI=0; mI<availableMessageCount; ++mI)
    {
        std::shared_ptr< OscMessage > oscMessage = receivedMessages[mI];
        
        for(auto lI=mMessageListeners.end() - 1; lI>=mMessageListeners.begin(); --lI)
        {
            auto messageListener = (*lI).lock();
            
            if( messageListener )
            {
				//std::cout << "receiver " << mName << " notify message " << mI  <<  "\n";
                
                messageListener->notify(oscMessage);
            }
            else
            {
                //std::cout << "erase\n";
                
                mMessageListeners.erase(lI);
            }
        }
    }
    
}

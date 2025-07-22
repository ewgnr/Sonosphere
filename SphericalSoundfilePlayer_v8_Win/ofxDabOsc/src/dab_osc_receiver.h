/** \file dab_osc_receiver.h
 */

#ifndef _dab_osc_receiver_h_
#define _dab_osc_receiver_h_

#include <osc/OscReceivedElements.h>
#include <osc/OscPacketListener.h>
#include <ip/UdpSocket.h>
//#include "ofEvent.h"
#include "ofThread.h"
#include "dab_osc_message.h"

#include <deque>
#include <memory>
#include <iostream>

namespace dab
{
    
#pragma mark OscListener definion
    
    class OscListener
    {
    public:
        virtual void notify( std::shared_ptr<OscMessage> pMessage ) = 0;
        
    protected:
    };
    
#pragma mark OscReceiver definition
    
    class OscReceiver : public osc::OscPacketListener
    {
        class ReceiverThread : public ofThread
        {
        public:
            ReceiverThread( OscReceiver* pReceiver );
            void threadedFunction();
            
        protected:
            OscReceiver* mReceiver;
        };
        
        class ListenerThread : public ofThread
        {
        public:
            ListenerThread( OscReceiver* pReceiver );
            void setUpdateInterval(unsigned int pUpdateInterval);
            void threadedFunction();
            
        protected:
            static unsigned int sUpdateInterval;
            
            OscReceiver* mReceiver;
            unsigned int mUpdateInterval;
        };
        
    public:
        OscReceiver();
        OscReceiver( const std::string& pName, unsigned int pPort );
        ~OscReceiver();        
        
        void setup( const std::string& pName, unsigned int pPort );
        void setListenInterval(unsigned int pListenInterval);
        void setMaxMessageCount(int pMaxMessageCount);
        
        void registerOscListener( std::weak_ptr<OscListener> pListener );
        
        const std::string& name() const;
        unsigned int port() const;
        
        void start();
        void start(unsigned int pListenInterval);
        void stop();
        
        void ProcessMessage( const osc::ReceivedMessage& pMessage, const osc::IpEndpointName& pRemoteEndpoint );
        void flushMessages();
        
    protected:
        std::string mName;
        unsigned int mReceivingPort;
        int mMaxMessageCount;
        osc::UdpListeningReceiveSocket* mReceiveSocket;
        
        std::map< unsigned long, std::shared_ptr<MultiOscMessage> > mPendingMultiOscMessages;
        //std::vector< std::shared_ptr< OscMessage > > mReceivedMessages;
        std::deque< std::shared_ptr< OscMessage > > mReceivedMessages;
        //ofEvent< std::shared_ptr< OscMessage> > mReceivedMessageEvent;
        std::vector< std::weak_ptr<OscListener> > mMessageListeners;
        std::mutex mMutex;
        
        ReceiverThread* mReceiverThread;
        ListenerThread* mListenerThread;
        
        void updateListeners();
    };
    
    
};

#endif

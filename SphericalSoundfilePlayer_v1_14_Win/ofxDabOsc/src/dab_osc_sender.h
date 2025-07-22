/** \file dab_osc_sender.h
 */

#ifndef _dab_osc_sender_h_
#define _dab_osc_sender_h_

#include <iostream>
#include <deque>
#include <memory>
#include <mutex>
#include <osc/OscOutboundPacketStream.h>
#include <ip/UdpSocket.h>
#include "ofThread.h"
#include "dab_osc_message.h"

namespace dab
{
    
    //class osc::UdpTransmitSocket;
    
    class OscSender : public ofThread
    {
    public:
        OscSender( const std::string& pName, const std::string& pIpAddress, unsigned int pPort );
        ~OscSender();
        
        const std::string& name() const;
        const std::string& address() const;
        unsigned int port() const;
        
        void start();
        void stop();
        
        void send();
        void send( std::shared_ptr<OscMessage> pMessage );
        void queue( std::shared_ptr<OscMessage> pMessage );
        
    protected:
        void send( std::shared_ptr< osc::OutboundPacketStream > pPacketStream );
        void queue( std::shared_ptr< osc::OutboundPacketStream > pPacketStream );
        
        std::string mName;
        std::string mSendingIpAddress;
        unsigned int mSendingPort;
        osc::UdpTransmitSocket mTransmitSocket;
        std::deque< std::shared_ptr< osc::OutboundPacketStream > > mMessageQueue;
        
        void threadedFunction();

		std::mutex mMutex;
    };
    
};

#endif

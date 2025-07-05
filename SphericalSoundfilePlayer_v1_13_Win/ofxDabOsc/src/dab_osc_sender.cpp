/** \file dab_osc_sender.cpp
 */

#include "dab_osc_sender.h"
#include "dab_osc_utils.h"
#include <chrono>
#include <thread>

using namespace dab;

OscSender::OscSender( const std::string& pName, const std::string& pIpAddress, unsigned int pPort )
: mName(pName)
, mSendingIpAddress(pIpAddress)
, mSendingPort(pPort)
, mTransmitSocket( osc::IpEndpointName(pIpAddress.c_str(), pPort) )
{}

OscSender::~OscSender()
{}

const std::string&
OscSender::name() const
{
    return mName;
}

const std::string&
OscSender::address() const
{
    return mSendingIpAddress;
}

unsigned int
OscSender::port() const
{
    return mSendingPort;
}

void
OscSender::start()
{
    if( isThreadRunning() == false ) startThread();
}

void
OscSender::stop()
{
    if( isThreadRunning() == true ) stopThread();
}

void
OscSender::send()
{
    mMutex.lock();
    
    while( mMessageQueue.size() > 0 )
    {
        send( mMessageQueue.front() );
        mMessageQueue.pop_front();
    }
    
    mMutex.unlock();
}

void
OscSender::send( std::shared_ptr<OscMessage> pMessage )
{
    //std::cout << "send message " << *pMessage << "\n";
    
    try
    {
        OscUtils& oscUtils = OscUtils::get();
        
        unsigned int extendedMessagePacketSize = oscUtils.extendedMessagePacketSize(pMessage);
        if(  extendedMessagePacketSize <= oscUtils.maxOscPacketSize() ) // the entire osc message fits into a single udp packet
        {
            std::shared_ptr< osc::OutboundPacketStream > packetStream = oscUtils.createPacketStream( pMessage );
            send( packetStream );
        }
        else
        {
            std::shared_ptr<MultiOscMessage> multiOscMessage = oscUtils.splitMessage(pMessage);
            std::vector< std::shared_ptr<osc::OutboundPacketStream> > packetStreams = oscUtils.createPacketStreams(multiOscMessage);
            
            int packetStreamCount = packetStreams.size();
            for(int pI=0; pI<packetStreamCount; ++pI)
            {                
                send( packetStreams[pI] );
            }
            
            packetStreams.clear();
        }
    }
    catch( Exception& e)
    {
        throw e += Exception("OSC Error: failed to send osc message", __FILE__, __FUNCTION__, __LINE__ );
    }
}

void
OscSender::queue( std::shared_ptr<OscMessage> pMessage )
{
    mMutex.lock();
    
    try
    {
        OscUtils& oscUtils = OscUtils::get();
        
        unsigned int extendedMessagePacketSize = oscUtils.extendedMessagePacketSize(pMessage);
        if(  extendedMessagePacketSize <= oscUtils.maxOscPacketSize() ) // the entire osc message fits into a single udp packet
        {
            std::shared_ptr< osc::OutboundPacketStream > packetStream = oscUtils.createPacketStream( pMessage );
            queue( packetStream );
        }
        else
        {
            std::shared_ptr<MultiOscMessage> multiOscMessage = oscUtils.splitMessage(pMessage);
            std::vector< std::shared_ptr<osc::OutboundPacketStream> > packetStreams = oscUtils.createPacketStreams(multiOscMessage);
            
            int packetStreamCount = packetStreams.size();
            for(int pI=0; pI<packetStreamCount; ++pI)
            {
                queue( packetStreams[pI] );
            }
        }
    }
    catch( Exception& e)
    {
        throw e += Exception("OSC Error: failed to send osc message", __FILE__, __FUNCTION__, __LINE__ );
    }
    
    mMutex.unlock();
}

void
OscSender::send( std::shared_ptr< osc::OutboundPacketStream > pPacketStream )
{
    mTransmitSocket.Send( pPacketStream->Data(), pPacketStream->Size() );
}

void
OscSender::queue( std::shared_ptr< osc::OutboundPacketStream > pPacketStream )
{
    mMessageQueue.push_back(pPacketStream);
}

void
OscSender::threadedFunction()
{
    while(isThreadRunning())
    {
        send();
    
		std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

//void
//OscSender::send( osc::OutboundPacketStream& pPacketStream )
//{
//    std::cout << "send\n";
//    
//    mTransmitSocket.Send( pPacketStream.Data(), pPacketStream.Size() );
//}

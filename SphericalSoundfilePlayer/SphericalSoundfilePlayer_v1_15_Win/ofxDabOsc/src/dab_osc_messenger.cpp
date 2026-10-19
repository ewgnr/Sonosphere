/** \file dab_com_messenger.cpp
 */

#include "dab_osc_messenger.h"
#include "dab_osc_sender.h"
#include "dab_osc_receiver.h"

#include "ofUtils.h"

using namespace dab;

OscMessenger::OscMessenger()
{}

OscMessenger::~OscMessenger()
{
	removeSenders();
	removeReceivers();
}

const std::vector< std::string >&
OscMessenger::senderNames() const
{
	return mSenders.keys();
}

bool
OscMessenger::checkSender( const std::string& pSenderName ) const
{
	return mSenders.contains(pSenderName);
}

std::shared_ptr<OscSender>
OscMessenger::sender( const std::string& pSenderName )
{
    try
    {
        return mSenders[pSenderName];
    }
    catch (Exception& e)
    {
        e += Exception("OSC Error: sender with name " + pSenderName + " not found", __FILE__, __FUNCTION__, __LINE__ );
        throw e;
    }
}

const std::string&
OscMessenger::senderIP( const std::string& pSenderName ) const
{
    try
    {
        return mSenders[pSenderName]->address();
    }
    catch (Exception& e)
    {
        e += Exception("OSC Error: sender with name " + pSenderName + " not found", __FILE__, __FUNCTION__, __LINE__ );
        throw e;
    }
}

unsigned int
OscMessenger::senderPort( const std::string& pSenderName ) const
{
    try
    {
        return mSenders[pSenderName]->port();
    }
    catch (Exception& e)
    {
        e += Exception("OSC Error: sender with name " + pSenderName + " not found", __FILE__, __FUNCTION__, __LINE__ );
        throw e;
    }
}

std::shared_ptr<OscSender>
OscMessenger::addSender( std::shared_ptr<OscSender> pSender)
{
    try
    {
        mSenders.insert( pSender->name(), pSender );
        return pSender;
    }
    catch (Exception& e)
    {
        e += Exception("OSC Error: sender with name " + pSender->name() + " already exists", __FILE__, __FUNCTION__, __LINE__ );
        throw e;
    }
}

std::shared_ptr<OscSender>
OscMessenger::createSender(const std::string& pSenderName, const std::string& pReceiverIP, unsigned int pReceiverPort)
{
	if( mSenders.contains( pSenderName ) ) throw Exception( "OSC ERROR: sender with name " + pSenderName + " aleady registered", __FILE__, __FUNCTION__, __LINE__ );

    std::shared_ptr<OscSender> _sender( new OscSender( pSenderName, pReceiverIP, pReceiverPort ) );
    mSenders.insert( pSenderName, _sender );
    
    return _sender;
}

void
OscMessenger::removeSender(const std::string& pSenderName)
{
    try
    {
        mSenders.remove(pSenderName);
    }
    catch (Exception& e)
    {
        e += Exception("OSC ERROR: sender with name " + pSenderName + " not found", __FILE__, __FUNCTION__, __LINE__ );
        throw e;
    }
}

void
OscMessenger::removeSenders()
{
    mSenders.clear();
}

const std::vector< std::string >&
OscMessenger::receiverNames() const
{
	return mReceivers.keys();
}

bool
OscMessenger::checkReceiver( const std::string& pReceiverName ) const
{
	return mReceivers.contains(pReceiverName);
}

std::shared_ptr<OscReceiver>
OscMessenger::receiver( const std::string& pReceiverName )
{
    try
    {
        return mReceivers[pReceiverName];
    }
    catch (Exception& e)
    {
        e += Exception("OSC ERROR: receiver with name " + pReceiverName + " not found", __FILE__, __FUNCTION__, __LINE__ );
        throw e;
    }
}

unsigned int
OscMessenger::receiverPort( const std::string& pReceiverName ) const
{
    try
    {
        return mReceivers[pReceiverName]->port();
    }
    catch (Exception& e)
    {
        e += Exception("OSC ERROR: receiver with name " + pReceiverName + " not found", __FILE__, __FUNCTION__, __LINE__ );
        throw e;
    }
}

std::shared_ptr<OscReceiver>
OscMessenger::addReceiver( std::shared_ptr<OscReceiver> pReceiver)
{
    try
    {
        mReceivers.insert( pReceiver->name(), pReceiver );
        pReceiver->start();
        
        return pReceiver;
    }
    catch (Exception& e)
    {
        e += Exception("OSC ERROR: receiver with name " + pReceiver->name() + " already exists", __FILE__, __FUNCTION__, __LINE__ );
        throw e;
    }
}

std::shared_ptr<OscReceiver>
OscMessenger::createReceiver(const std::string& pReceiverName, unsigned int pListeningPort)
{
	if( mReceivers.contains( pReceiverName ) ) throw Exception( "OSC ERROR: receiver with name " + pReceiverName + " aleady registered", __FILE__, __FUNCTION__, __LINE__ );
		// std::cout << "new receiver " <<  pReceiverName.toStdString() << "\n";
		
    std::shared_ptr<OscReceiver> _receiver( new OscReceiver( pReceiverName, pListeningPort ) );
    mReceivers.insert( pReceiverName, _receiver );
    _receiver->start();
    
    return _receiver;
}

void
OscMessenger::removeReceiver(const std::string& pReceiverName)
{
    try
    {
        mReceivers.remove(pReceiverName);
    }
    catch (Exception& e)
    {
        e += Exception("OSC ERROR: receiver with name " + pReceiverName + " already exists", __FILE__, __FUNCTION__, __LINE__ );
        throw e;
    }
}

void
OscMessenger::removeReceivers()
{
    mReceivers.clear();
}

void
OscMessenger::registerListener( const std::string& pReceiverName, std::shared_ptr<OscListener> pListener )
{
    try
    {
        mReceivers[pReceiverName]->registerOscListener(pListener);
    }
    catch (Exception& e)
    {
        e += Exception("OSC ERROR: failed to register message listener for receiver with name " + pReceiverName, __FILE__, __FUNCTION__, __LINE__ );
        throw e;
    }
}

void
OscMessenger::send( std::shared_ptr<OscMessage> pMessage)
{
    int senderCount = mSenders.size();
    for(int sI=0; sI<senderCount; ++sI)
    {
        try
        {
            mSenders[sI]->send(pMessage);
        }
        catch(Exception& e)
        {
            std::cout << "Messenger::send( const OscMessage& pMessage) failed\n";
            std::cout << e << "\n";
        }
    }
}

void
OscMessenger::send( const std::string& pSenderName, std::shared_ptr<OscMessage> pMessage)
{
    try
    {
        mSenders[pSenderName]->send(pMessage);
    }
    catch (Exception& e)
    {
        e += Exception("OSC ERROR: failed to send message through sender with name " + pSenderName, __FILE__, __FUNCTION__, __LINE__ );
        throw e;
    }
}

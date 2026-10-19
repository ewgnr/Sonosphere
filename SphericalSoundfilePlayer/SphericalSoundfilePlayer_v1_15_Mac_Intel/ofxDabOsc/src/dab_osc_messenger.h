/** \file dab_osc_messenger.h
 */

#ifndef _dab_osc_messenger_h_
#define _dab_osc_messenger_h_

#include "dab_exception.h"
#include "dab_osc_message.h"
#include "dab_index_map.h"
#include <map>

namespace dab
{


class OscSender;
class OscReceiver;
class OscListener;

class OscMessenger
{
public:
    
    /**
     \brief create messenger
     */
    OscMessenger();
    
    /**
     \brief destructor
     */
    virtual ~OscMessenger();
    
    /**
     \brief return sender names
     \return sender names
     */
    const std::vector< std::string >& senderNames() const;
    
    /**
     \brief check sender
     \param pSenderName name of sender
     \return true if sender is found, false otherwise
     */
    bool checkSender( const std::string& pSenderName ) const;
    
    /**
     \brief retrieve sender
     \param pSenderName name of sender
     \return sender
     \exception Exception sender not found
     */
    std::shared_ptr<OscSender> sender( const std::string& pSenderName );
    
    /**
     \brief return sender ip
     \param pSenderName name of sender
     \returns sender ip
     \exception Exception sender not found
     */
    const std::string& senderIP( const std::string& pSenderName ) const;
    
    /**
     \brief return sender ip
     \param pSenderName name of sender
     \returns sender ip
     \exception ComException sender not found
     */
    unsigned int senderPort( const std::string& pSenderName ) const;
    
    /**
     \brief add sender
     \param pSender sender
     \returns sender
     \exception Exception Exception sender with identical name already registered
     */
    std::shared_ptr<OscSender> addSender( std::shared_ptr<OscSender> pSender);
    
    /**
     \brief create network sender
     \param pSenderName name of network sender
     \param pReceiverIP IP address of receiver
     \param pReceiverPort port of receiver
     \returns sender
     \exception Exception failed to create network sender
     */
    std::shared_ptr<OscSender> createSender(const std::string& pSenderName, const std::string& pReceiverIP, unsigned int pReceiverPort);
    
    /**
     \brief remove sender
     \param pSenderName name of sender
     \exception Exception failed to remove sender
     */
    virtual void removeSender(const std::string& pSenderName);
    
    /**
     \brief remove all senders
     */
    virtual void removeSenders();
    
    /**
     \brief return receiver names
     \return receiver names
     */
    const std::vector< std::string >& receiverNames() const;
    
    /**
     \brief check receiver
     \param pReceiverName name of receiver
     \return true if receiver is found, false otherwise
     */
    bool checkReceiver( const std::string& pReceiverName ) const;
    
    /**
     \brief retrieve receiver
     \param pReceiverName name of receiver
     \return receiver
     \exception Exception receiver not found
     */
    std::shared_ptr<OscReceiver> receiver( const std::string& pReceiverName );
    
    /**
     \brief return receiver port
     \param pReceiverName name of receiver
     \returns receiver port
     \exception Exception receiver not found
     */
    unsigned int receiverPort( const std::string& pReceiverName ) const;
    
    
    /**
     \brief add receiver
     \param pReceiver receiver
     \return receiver
     \exception Exception Exception receiver with identical name already registered
     */
    std::shared_ptr<OscReceiver> addReceiver( std::shared_ptr<OscReceiver> pReceiver);
    
    /**
     \brief create network receiver
     \param pReceiverName name of network receiver
     \param pListeningPort receiver listening port
     \return receiver
     \exception Exception failed to create network receiver
     */
    std::shared_ptr<OscReceiver> createReceiver(const std::string& pReceiverName, unsigned int pListeningPort);
    
    /**
     \brief remove receiver
     \param pReceiverName name of receiver
     \exception failed to remove receiver
     */
    virtual void removeReceiver(const std::string& pReceiverName);
    
    /**
     \brief remove all receivers
     */
    virtual void removeReceivers();
    
    /**
     \brief register message listener
     \param pReceiverName name of receiver
     \param pListener message listener
     \exception Exception receiver not found
     */
    void registerListener( const std::string& pReceiverName, std::shared_ptr<OscListener> pListener );
    
    /**
     \brief send message
     \param pMessage message
     \remarks sends the same message via all senders
     \exception failure in sending produces an error message in the programs verbose output
     */
    void send( std::shared_ptr<OscMessage> pMessage);
    
    /**
     \brief send message
     \param pSenderName name of sender
     \param pMessage message
     \exception ComException sender not found
     \remarks sends message via single specified sender
     \remarks failure in sending produces an error message in the programs verbose output
     */
    void send( const std::string& pSenderName, std::shared_ptr<OscMessage> pMessage);
    
protected:
    IndexMap<std::string, std::shared_ptr<OscSender> > mSenders; ///\brief message sender repository
    IndexMap<std::string, std::shared_ptr<OscReceiver> > mReceivers;  ///\brief message receiver repository
};


};

#endif

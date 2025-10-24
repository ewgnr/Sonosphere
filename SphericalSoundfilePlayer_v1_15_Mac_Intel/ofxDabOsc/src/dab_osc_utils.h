/** \file dab_osc_utils.h
 //
 // Extended Single Message:
 //
 // * address (string)
 // * 0: message type (char) , always 'e' at the moment
 // * 1: message array types (string)
 // * 2+: message arguments
 //
 // Start Multi Message:
 //
 // * address (string)
 // * 0: message type (char) , always 'm' at the moment
 // * 1: message id (int64)
 // * 2: message count (int32) > 1
 // * 3: message index (int32) = 0
 // * 4: message all array types (string)
 // * 5+: message all array arg sizes, size0 ... sizeN (int32)
 // * >=6+: message arguments
 //
 //
 // Next Multi Message:
 //
 // * address (string)
 // * 0: message type (char) , always 'm' at the moment
 // * 1: message id (int64)
 // * 2: message count (int32) > 1
 // * 3: message index (int32) > 0
 // * 4: message array types (string)
 // * 5+: message arguments
 */

#ifndef _dab_osc_utils_h_
#define _dab_osc_utils_h_

#include <iostream>
#include <map>
#include <memory>
#include <typeindex>
#include <osc/OscReceivedElements.h>
#include <osc/OscOutboundPacketStream.h>
#include "dab_singleton.h"
#include "dab_osc_types.h"
#include "dab_exception.h"

namespace dab
{
    
    class _OscArg;
    class OscMessage;
    class MultiOscMessage;
    
#pragma mark OscUtils definiton
    
    class OscUtils : public Singleton< OscUtils >
    {
    public:
        friend class Singleton< OscUtils >;
        
        unsigned int maxOscPacketSize() const;
        
        template< typename Type > OscType oscType( const Type& pType );
        template< typename Type > OscType oscType();
        template< typename Type > OscTag oscTag( const Type& pType );
        template< typename Type > OscTag oscTag();
        
        OscTag oscTag( OscType pType );
        OscTag oscArrayTag( OscType pType );
        const std::string& typeName( OscType pOscType );
        
        bool isConventionalMessage( const osc::ReceivedMessage& pOscMessage );
        bool isExtendedMessage( const osc::ReceivedMessage& pOscMessage );
        bool isMultiOscMessage( const osc::ReceivedMessage& pOscMessage );
        bool isStartMultiOscMessage( const osc::ReceivedMessage& pOscMessage );
        bool isNextMultiOscMessage( const osc::ReceivedMessage& pOscMessage );
        
        unsigned long messageId( const osc::ReceivedMessage& pOscMessage );
        
        unsigned int extendedMessagePacketSize( const _OscArg& pArgument ); // calculate number of bytes needed to store both header information and value content of argument in a single extended osc message
        unsigned int multiMessagePacketSize( const _OscArg& pArgument ); // calculate number of bytes needed to store both header information and value content of argument in a multi osc message
        unsigned int startMultiMessagePacketSize( const _OscArg& pArgument ); // calculate number of bytes needed to store both header information and value content of argument in a start multi osc message
        
        unsigned int extendedMessagePacketSize( std::shared_ptr<OscMessage> pMessage ); // calculate number of bytes required to send a single osc message
        
        std::shared_ptr<osc::OutboundPacketStream> createPacketStream( std::shared_ptr<OscMessage> pMessage );
        std::vector< std::shared_ptr<osc::OutboundPacketStream> > createPacketStreams( std::shared_ptr<MultiOscMessage> pMultiOscMessage );
        
        std::shared_ptr< OscMessage > createMessage( const osc::ReceivedMessage& pReceivedMessage );
        std::shared_ptr< MultiOscMessage > createMultiOscMessage( const osc::ReceivedMessage& pReceivedMessage );
        void updateMultiOscMessage( const osc::ReceivedMessage& pReceivedMessage, std::shared_ptr< MultiOscMessage > pMultiOscMssage );
        
        std::shared_ptr<MultiOscMessage> splitMessage( std::shared_ptr<OscMessage> pMessage );
        std::shared_ptr<OscMessage> mergeMessage( std::shared_ptr<MultiOscMessage> pMultiMessage );
        
        //    std::string getArrayTypeTags( const osc::ReceivedMessage& pOscMessage ) throw (Exception);
        
    protected:
		static const unsigned int sMaxOscMessageSize;
		static const unsigned int sEmptySingleMessageHeaderSize;
		static const unsigned int sEmptyMultiMessageHeaderSize;
		static const unsigned int sEmptyStartMultiMessageHeaderSize;
		static const unsigned int sExtendedMessageArrayHeaderSize;
		static const unsigned int sMultiMessageArrayHeaderSize;
		static const unsigned int sStartMultiMessageArrayHeaderSize;
        
        static bool sInitialized;
        static std::map< std::type_index, OscType > sDataTypeOscTypeMap;
        static std::map< std::type_index, OscTag > sDataTypeOscTagMap;
        static std::map< OscType, OscTag > sOscTypeOscTagMap;
        static std::map< OscType, OscTag > sOscArrayTypeOscTagMap;
        static std::map< OscType, std::string > sOscTypeNames;
        
        static void initialize();
        
        
        OscUtils();
        ~OscUtils();
    };
    
#pragma mark OscUtils implementation
    
    template< typename Type >
    OscType
    OscUtils::oscType( const Type& pType )
    {
        std::type_index typeIndex = std::type_index(typeid(pType));
        
        if( sDataTypeOscTypeMap.find(typeIndex) == sDataTypeOscTypeMap.end() ) return OSC_TYPE_NONE;
        return sDataTypeOscTypeMap[typeIndex];
    }
    
    template< typename Type >
    OscType
    OscUtils::oscType()
    {
        std::type_index typeIndex = std::type_index(typeid(Type));
        
        //std::cout << "OscUtils::oscType() typeIndex " << typeIndex.name() << "\n";
        
        if( sDataTypeOscTypeMap.find(typeIndex) == sDataTypeOscTypeMap.end() ) return OSC_TYPE_NONE;
        return sDataTypeOscTypeMap[typeIndex];
    }
    
    template< typename Type >
    OscTag
    OscUtils::oscTag( const Type& pType )
    {
        std::type_index typeIndex = std::type_index(typeid(pType));
        
        if( sDataTypeOscTagMap.find(typeIndex) == sDataTypeOscTagMap.end() ) return OSC_TAG_NONE;
        return sDataTypeOscTagMap[typeIndex];
    }
    
    template< typename Type >
    OscTag
    OscUtils::oscTag()
    {
        std::type_index typeIndex = std::type_index(typeid(Type));
        
        if( sDataTypeOscTagMap.find(typeIndex) == sDataTypeOscTagMap.end() ) return OSC_TAG_NONE;
        return sDataTypeOscTagMap[typeIndex];
    }
    
};

#endif

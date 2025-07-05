/** \file dab_osc_utils.cpp
 */

#include "ofUtils.h"
#include "dab_osc_utils.h"
#include "dab_osc_arg.h"
#include "dab_osc_message.h"
#include <algorithm>

using namespace dab;

const unsigned int OscUtils::sMaxOscMessageSize = 4096; // 2048;
const unsigned int OscUtils::sEmptySingleMessageHeaderSize = 10 + 1 + 1 // empty argTags + ,
+ 1 // ,
+ sizeof(char) + 1 // 'e' + argTag
+ 1 + 1; // empty arrayTags + argTag
const unsigned int OscUtils::sEmptyMultiMessageHeaderSize = 10 + 1 + 1 // empty argTags + ,
+ sizeof(char) + 1 // 'm' + argTag
+ sizeof(int64_t) + 1 // message id + argTag
+ sizeof(int32_t) + 1 // message count + argTag
+ sizeof(int32_t) + 1 // message index + argTag
+ 1 + 1; // empty arrayTags + argTag
const unsigned int OscUtils::sEmptyStartMultiMessageHeaderSize = 10 + 1 + 1 // empty argTags + ,
+ sizeof(char) + 1 // 'm' + argTag
+ sizeof(int64_t) + 1 // message id + argTag
+ sizeof(int32_t) + 1 // message count + argTag
+ sizeof(int32_t) + 1 // message index + argTag
+ 1 + 1; // empty arrayTags + argTag

const unsigned int OscUtils::sExtendedMessageArrayHeaderSize = 10 + 1 + 1; // argTags + arrayTags
const unsigned int OscUtils::sMultiMessageArrayHeaderSize = 10 + 1 + 1; // argTags + arrayTags
const unsigned int OscUtils::sStartMultiMessageArrayHeaderSize = 10 + 1 + 1; // argTags + arrayTags

bool OscUtils::sInitialized = false;
std::map< std::type_index, OscType > OscUtils::sDataTypeOscTypeMap;
std::map< std::type_index, OscTag > OscUtils::sDataTypeOscTagMap;
std::map< OscType, OscTag > OscUtils::sOscTypeOscTagMap;
std::map< OscType, OscTag > OscUtils::sOscArrayTypeOscTagMap;
std::map< OscType, std::string > OscUtils::sOscTypeNames;

OscUtils::OscUtils()
{
    if( sInitialized == false ) initialize();
}

OscUtils::~OscUtils()
{}

void
OscUtils::initialize()
{
    // ambiguous cases:
    // string & symbol : std::string
    // midi & int32 : int32
    // bool, true, false
    
    //auto stringVar = "string";
    
    sDataTypeOscTypeMap[ std::type_index(typeid(bool)) ] = OSC_TYPE_TRUE;
    sDataTypeOscTypeMap[ std::type_index(typeid(char)) ] = OSC_TYPE_CHAR;
    sDataTypeOscTypeMap[ std::type_index(typeid(int)) ] = OSC_TYPE_INT32;
    sDataTypeOscTypeMap[ std::type_index(typeid(int32_t)) ] = OSC_TYPE_INT32;
    //sDataTypeOscTypeMap[ std::type_index(typeid(int32_t)) ] = OSC_TYPE_RGBA_COLOR;
    sDataTypeOscTypeMap[ std::type_index(typeid(long)) ] = OSC_TYPE_INT64;
    sDataTypeOscTypeMap[ std::type_index(typeid(int64_t)) ] = OSC_TYPE_INT64;
    //sDataTypeOscTypeMap[ std::type_index(typeid(int64_t)) ] = OSC_TYPE_TIMETAG;
    sDataTypeOscTypeMap[ std::type_index(typeid(float)) ] = OSC_TYPE_FLOAT;
    sDataTypeOscTypeMap[ std::type_index(typeid(double)) ] = OSC_TYPE_DOUBLE;
    sDataTypeOscTypeMap[ std::type_index(typeid(std::string)) ] = OSC_TYPE_STRING;
    sDataTypeOscTypeMap[ std::type_index(typeid(char*)) ] = EXT_TYPE_ARG_CHAR_ARRAY;
    sDataTypeOscTypeMap[ std::type_index(typeid(std::string)) ] = OSC_TYPE_STRING;
    sDataTypeOscTypeMap[ std::type_index(typeid(int*)) ] = EXT_TYPE_ARG_INT32_ARRAY;
    sDataTypeOscTypeMap[ std::type_index(typeid(int32_t*)) ] = EXT_TYPE_ARG_INT32_ARRAY;
    sDataTypeOscTypeMap[ std::type_index(typeid(long*)) ] = EXT_TYPE_ARG_INT64_ARRAY;
    sDataTypeOscTypeMap[ std::type_index(typeid(int64_t*)) ] = EXT_TYPE_ARG_INT64_ARRAY;
    sDataTypeOscTypeMap[ std::type_index(typeid(float*)) ] = EXT_TYPE_ARG_FLOAT_ARRAY;
    sDataTypeOscTypeMap[ std::type_index(typeid(double*)) ] = EXT_TYPE_ARG_DOUBLE_ARRAY;
    
    sDataTypeOscTagMap[ std::type_index(typeid(bool)) ] = OSC_TAG_TRUE;
    sDataTypeOscTagMap[ std::type_index(typeid(char)) ] = OSC_TAG_CHAR;
    sDataTypeOscTagMap[ std::type_index(typeid(int)) ] = OSC_TAG_INT32;
    sDataTypeOscTagMap[ std::type_index(typeid(int32_t)) ] = OSC_TAG_INT32;
    //sDataTypeOscTagMap[ std::type_index(typeid(int32_t)) ] = OSC_TAG_RGBA_COLOR;
    sDataTypeOscTagMap[ std::type_index(typeid(long)) ] = OSC_TAG_INT64;
    sDataTypeOscTagMap[ std::type_index(typeid(int64_t)) ] = OSC_TAG_INT64;
    //sDataTypeOscTagMap[ std::type_index(typeid(int64_t)) ] = OSC_TAG_TIMETAG;
    sDataTypeOscTagMap[ std::type_index(typeid(float)) ] = OSC_TAG_FLOAT;
    sDataTypeOscTagMap[ std::type_index(typeid(double)) ] = OSC_TAG_DOUBLE;
    sDataTypeOscTagMap[ std::type_index(typeid(std::string)) ] = OSC_TAG_STRING;
    sDataTypeOscTagMap[ std::type_index(typeid(char*)) ] = OSC_TAG_BLOB;
    sDataTypeOscTagMap[ std::type_index(typeid(std::string)) ] = OSC_TAG_STRING;
    sDataTypeOscTagMap[ std::type_index(typeid(int*)) ] = EXT_TAG_ARG_INT32_ARRAY;
    sDataTypeOscTagMap[ std::type_index(typeid(int32_t*)) ] = EXT_TAG_ARG_INT32_ARRAY;
    sDataTypeOscTagMap[ std::type_index(typeid(long*)) ] = EXT_TAG_ARG_INT64_ARRAY;
    sDataTypeOscTagMap[ std::type_index(typeid(int64_t*)) ] = EXT_TAG_ARG_INT64_ARRAY;
    sDataTypeOscTagMap[ std::type_index(typeid(float*)) ] = EXT_TAG_ARG_FLOAT_ARRAY;
    sDataTypeOscTagMap[ std::type_index(typeid(double*)) ] = EXT_TAG_ARG_DOUBLE_ARRAY;
    
    sOscTypeOscTagMap[OSC_TYPE_TRUE] = OSC_TAG_TRUE;
    sOscTypeOscTagMap[OSC_TYPE_FALSE] = OSC_TAG_FALSE;
    sOscTypeOscTagMap[OSC_TYPE_CHAR] = OSC_TAG_CHAR;
    sOscTypeOscTagMap[OSC_TYPE_INT32] = OSC_TAG_INT32;
    sOscTypeOscTagMap[OSC_TYPE_INT64] = OSC_TAG_INT64;
    sOscTypeOscTagMap[OSC_TYPE_FLOAT] = OSC_TAG_FLOAT;
    sOscTypeOscTagMap[OSC_TYPE_DOUBLE] = OSC_TAG_DOUBLE;
    sOscTypeOscTagMap[OSC_TYPE_STRING] = OSC_TAG_STRING;
    sOscTypeOscTagMap[OSC_TYPE_SYMBOL] = OSC_TAG_SYMBOL;
    sOscTypeOscTagMap[OSC_TYPE_MIDI_MESSAGE] = OSC_TAG_MIDI_MESSAGE;
    sOscTypeOscTagMap[OSC_TYPE_TRIGGER] = OSC_TAG_TRIGGER;
    sOscTypeOscTagMap[OSC_TYPE_TIMETAG] = OSC_TAG_TIMETAG;
    sOscTypeOscTagMap[OSC_TYPE_BLOB] = OSC_TAG_BLOB;
    sOscTypeOscTagMap[OSC_TYPE_BUNDLE] = OSC_TAG_BUNDLE;
    sOscTypeOscTagMap[OSC_TYPE_RGBA_COLOR] = OSC_TAG_RGBA_COLOR;
    sOscTypeOscTagMap[OSC_TYPE_ARRAY_BEGIN] = OSC_TAG_ARRAY_BEGIN;
    sOscTypeOscTagMap[OSC_TYPE_ARRAY_END] = OSC_TAG_ARRAY_END;
    sOscTypeOscTagMap[EXT_TYPE_ARG_CHAR_ARRAY] = OSC_TAG_BLOB;
    sOscTypeOscTagMap[EXT_TYPE_ARG_INT32_ARRAY] = OSC_TAG_BLOB;
    sOscTypeOscTagMap[EXT_TYPE_ARG_INT64_ARRAY] = OSC_TAG_BLOB;
    sOscTypeOscTagMap[EXT_TYPE_ARG_FLOAT_ARRAY] = OSC_TAG_BLOB;
    sOscTypeOscTagMap[EXT_TYPE_ARG_DOUBLE_ARRAY] = OSC_TAG_BLOB;
    sOscTypeOscTagMap[OSC_TYPE_NONE] = OSC_TAG_NONE;
    
    sOscArrayTypeOscTagMap[OSC_TYPE_STRING] = OSC_TAG_STRING;
    sOscArrayTypeOscTagMap[OSC_TYPE_SYMBOL] = OSC_TAG_SYMBOL;
    sOscArrayTypeOscTagMap[EXT_TYPE_ARG_CHAR_ARRAY] = EXT_TAG_ARG_CHAR_ARRAY;
    sOscArrayTypeOscTagMap[EXT_TYPE_ARG_INT32_ARRAY] = EXT_TAG_ARG_INT32_ARRAY;
    sOscArrayTypeOscTagMap[EXT_TYPE_ARG_INT64_ARRAY] = EXT_TAG_ARG_INT64_ARRAY;
    sOscArrayTypeOscTagMap[EXT_TYPE_ARG_FLOAT_ARRAY] = EXT_TAG_ARG_FLOAT_ARRAY;
    sOscArrayTypeOscTagMap[EXT_TYPE_ARG_DOUBLE_ARRAY] = EXT_TAG_ARG_DOUBLE_ARRAY;
    
    sOscTypeNames[OSC_TYPE_TRUE] = "OSC_TYPE_TRUE";
    sOscTypeNames[OSC_TYPE_FALSE] = "OSC_TYPE_FALSE";
    sOscTypeNames[OSC_TYPE_CHAR] = "OSC_TYPE_CHAR";
    sOscTypeNames[OSC_TYPE_INT32] = "OSC_TYPE_INT32";
    sOscTypeNames[OSC_TYPE_INT64] = "OSC_TYPE_INT64";
    sOscTypeNames[OSC_TYPE_FLOAT] = "OSC_TYPE_FLOAT";
    sOscTypeNames[OSC_TYPE_DOUBLE] = "OSC_TYPE_DOUBLE";
    sOscTypeNames[OSC_TYPE_STRING] = "OSC_TYPE_STRING";
    sOscTypeNames[OSC_TYPE_SYMBOL] = "OSC_TYPE_SYMBOL";
    sOscTypeNames[OSC_TYPE_MIDI_MESSAGE] = "OSC_TYPE_MIDI_MESSAGE";
    sOscTypeNames[OSC_TYPE_TRIGGER] = "OSC_TYPE_TRIGGER";
    sOscTypeNames[OSC_TYPE_TIMETAG] = "OSC_TYPE_TIMETAG";
    sOscTypeNames[OSC_TYPE_BLOB] = "OSC_TYPE_BLOB";
    sOscTypeNames[OSC_TYPE_BUNDLE] = "OSC_TYPE_BUNDLE";
    sOscTypeNames[OSC_TYPE_RGBA_COLOR] = "OSC_TYPE_RGBA_COLOR";
    sOscTypeNames[OSC_TYPE_ARRAY_BEGIN] = "OSC_TYPE_ARRAY_BEGIN";
    sOscTypeNames[OSC_TYPE_ARRAY_END] = "OSC_TYPE_ARRAY_END";
    sOscTypeNames[EXT_TYPE_ARG_CHAR_ARRAY] = "EXT_TYPE_ARG_CHAR_ARRAY";
    sOscTypeNames[EXT_TYPE_ARG_INT32_ARRAY] = "EXT_TYPE_ARG_INT32_ARRAY";
    sOscTypeNames[EXT_TYPE_ARG_INT64_ARRAY] = "EXT_TYPE_ARG_INT64_ARRAY";
    sOscTypeNames[EXT_TYPE_ARG_FLOAT_ARRAY] = "EXT_TYPE_ARG_FLOAT_ARRAY";
    sOscTypeNames[EXT_TYPE_ARG_DOUBLE_ARRAY] = "EXT_TYPE_ARG_DOUBLE_ARRAY";
    sOscTypeNames[OSC_TYPE_NONE] = "OSC_TYPE_NONE";
    
    sInitialized = true;
}

unsigned int
OscUtils::maxOscPacketSize() const
{
    return sMaxOscMessageSize;
}

OscTag
OscUtils::oscTag( OscType pType )
{
    return sOscTypeOscTagMap[pType];
}

OscTag
OscUtils::oscArrayTag( OscType pType )
{
    return sOscArrayTypeOscTagMap[pType];
}

const std::string&
OscUtils::typeName( OscType pOscType )
{
    return sOscTypeNames[pOscType];
}

bool
OscUtils::isConventionalMessage( const osc::ReceivedMessage& pOscMessage )
{
    if( pOscMessage.ArgumentCount() < 2 ) return true;

    osc::ReceivedMessage::const_iterator argIter = pOscMessage.ArgumentsBegin();
    
    if( argIter->IsChar() == false) return true;
    
    char charArg = argIter->AsCharUnchecked();
    
    if( charArg != 'e' && charArg != 'm' ) return true;
    
    return false;
}

bool
OscUtils::isExtendedMessage( const osc::ReceivedMessage& pOscMessage )
{
    if( pOscMessage.ArgumentCount() < 2 ) return false;
    
    osc::ReceivedMessage::const_iterator argIter = pOscMessage.ArgumentsBegin();
    
    if( argIter->IsChar() == false || (argIter)->AsCharUnchecked() != 'e') return false; // message type
    if( (++argIter)->IsString() == false ) return false; // array tags
    
    return true;
}

bool
OscUtils::isMultiOscMessage( const osc::ReceivedMessage& pOscMessage )
{
    if( pOscMessage.ArgumentCount() < 4 ) return false;

    osc::ReceivedMessage::const_iterator argIter = pOscMessage.ArgumentsBegin();
    if( argIter->IsChar() == false || argIter->AsCharUnchecked() != 'm' ) return false; // message type
    if( (++argIter)->IsInt64() == false ) return false; // message id
    if( (++argIter)->IsInt32() == false ) return false; // message count
    if( (++argIter)->IsInt32() == false ) return false; // message index
    if( (++argIter)->IsString() == false ) return false; // array tags
        
    return true;
}

bool
OscUtils::isStartMultiOscMessage( const osc::ReceivedMessage& pOscMessage )
{
    if( isMultiOscMessage(pOscMessage) == false ) return false;
    if( pOscMessage.ArgumentCount() < 5 ) return false;
    
    osc::ReceivedMessage::const_iterator argIter = pOscMessage.ArgumentsBegin();
    
    if( argIter->IsChar() == false || argIter->AsCharUnchecked() != 'm' ) return false; // message type
    if( (++argIter)->IsInt64() == false ) return false; // message id
    if( (++argIter)->IsInt32() == false ) return false; // message count
    if( (++argIter)->IsInt32() == false ) return false; // message index
    if( argIter->AsInt32Unchecked() != 0 ) return false; // message index
    if( (++argIter)->IsString() == false ) return false; // array tags

    
    std::string arrayTags = argIter->AsStringUnchecked();
    unsigned int arrayCount = arrayTags.size();

    if( pOscMessage.ArgumentCount() < 5 + arrayCount ) return false; // too few arguments
    
    for(int aI=0; aI<arrayCount; ++aI) // array sizes
    {
        if( (++argIter)->IsInt32() == false ) return false;
    }
    
    return true;
}

bool
OscUtils::isNextMultiOscMessage( const osc::ReceivedMessage& pOscMessage )
{
    if( isMultiOscMessage(pOscMessage) == false ) return false;
    
    if( pOscMessage.ArgumentCount() < 5 ) return false;
    
    osc::ReceivedMessage::const_iterator argIter = pOscMessage.ArgumentsBegin();
    
    if( argIter->IsChar() == false || argIter->AsCharUnchecked() != 'm' ) return false; // message type
    if( (++argIter)->IsInt64() == false ) return false; // message id
    if( (++argIter)->IsInt32() == false ) return false; // message count
    if( (++argIter)->IsInt32() == false ) return false; // message index
    if( argIter->AsInt32Unchecked() == 0 ) return false; // message index
    if( (++argIter)->IsString() == false ) return false; // array tags
    
    return true;
}

unsigned long
OscUtils::messageId( const osc::ReceivedMessage& pOscMessage )
{
    osc::ReceivedMessage::const_iterator argIter = pOscMessage.ArgumentsBegin();
    
    if( pOscMessage.ArgumentCount() < 5 || argIter->IsChar() == false || argIter->AsCharUnchecked() != 'm' || (++argIter)->IsInt64() == false ) throw Exception("OSC Error: ReceivedMessage is not a part of a MultiOscMessage", __FILE__, __FUNCTION__, __LINE__);
    
    return argIter->AsInt64Unchecked();
}

unsigned int
OscUtils::extendedMessagePacketSize( const _OscArg& pArgument )
{
    int argumentByteCount = pArgument.byteCount();
    int headerByteSize = 1; // type tag
    
    if( pArgument.valueCount() > 1 )  headerByteSize += 1; // array tag
    
    return argumentByteCount + headerByteSize;
}

unsigned int
OscUtils::multiMessagePacketSize( const _OscArg& pArgument )
{
    int argumentByteCount = pArgument.byteCount();
    int headerByteSize = 1; // type tag
    
    if( pArgument.valueCount() > 1 )  headerByteSize += 1; // array tag
    
    return argumentByteCount + headerByteSize;
}

unsigned int
OscUtils::startMultiMessagePacketSize( const _OscArg& pArgument )
{
    int argumentByteCount = pArgument.byteCount();
    int headerByteSize = 1; // type tag
    
    if( pArgument.valueCount() > 1  )
    {
        headerByteSize += sizeof(int); // array size
        headerByteSize += 1; // array tag
    }
    
    return argumentByteCount + headerByteSize;
}

unsigned int
OscUtils::extendedMessagePacketSize( std::shared_ptr<OscMessage> pMessage )
{
    int messageSize = sEmptySingleMessageHeaderSize;
    
    const std::string& messageAddress = pMessage->address();
    messageSize += messageAddress.size();
    
    const std::vector<_OscArg*>& messageArgs = pMessage->arguments();
    unsigned int argCount = messageArgs.size();
    
    for(int aI=0; aI<argCount; ++aI)
    {
        messageSize += extendedMessagePacketSize( *(messageArgs[aI]) );
    }
    
    return messageSize;
}

std::shared_ptr< OscMessage >
OscUtils::createMessage( const osc::ReceivedMessage& pReceivedMessage )
{
    if( isMultiOscMessage(pReceivedMessage) ) throw ("OSC Error: ReceivedMessage forms part of a MultiOscMessage", __FILE__, __FUNCTION__, __LINE__);
    
    try
    {
        
        std::string messageAddress = pReceivedMessage.AddressPattern();
        
        std::shared_ptr<OscMessage> oscMessage( new OscMessage(messageAddress) );
        osc::ReceivedMessage::const_iterator arg = pReceivedMessage.ArgumentsBegin();
    
        if( isConventionalMessage(pReceivedMessage) ) // conventional osc message
        {
            //std::cout << "conventional message\n";
            
            //arg++; // message type
            
            while( arg != pReceivedMessage.ArgumentsEnd() )
            {
                if( arg->IsBool() ) oscMessage->add((arg++)->AsBoolUnchecked() );
                if( arg->IsChar() ) oscMessage->add((arg++)->AsCharUnchecked() );
                else if( arg->IsInt32() ) oscMessage->add((arg++)->AsInt32Unchecked() );
                else if( arg->IsInt64() ) oscMessage->add((arg++)->AsInt64Unchecked() );
                else if( arg->IsFloat() ) oscMessage->add((arg++)->AsFloatUnchecked() );
                else if( arg->IsDouble() ) oscMessage->add((arg++)->AsDoubleUnchecked() );
                else if( arg->IsString() )
                {
                    std::string argString = (arg++)->AsStringUnchecked();
                    oscMessage->add(argString);
                }
                else if( arg->IsSymbol() )
                {
                    std::string argString = (arg++)->AsStringUnchecked();
                    oscMessage->add(argString, OSC_TYPE_SYMBOL);
                }
                else if( arg->IsMidiMessage() ) oscMessage->add( (int)(arg++)->AsMidiMessageUnchecked(), OSC_TYPE_MIDI_MESSAGE );
                else if( arg->IsTimeTag() ) oscMessage->add( (long)(arg++)->AsTimeTagUnchecked(), OSC_TYPE_TIMETAG );
                else if( arg->IsBlob() )
                {
                    char * dataPtr;
                    osc::osc_bundle_element_size_t len = 0;
                    arg->AsBlobUnchecked((const void*&)dataPtr, len);
                    oscMessage->add( dataPtr, len, OSC_TYPE_BLOB);
                    delete [] dataPtr;
                }
                else if( arg->IsRgbaColor() ) oscMessage->add((int)(arg++)->AsRgbaColorUnchecked(), OSC_TYPE_RGBA_COLOR );
            }
        }
        else // extended osc message
        {
            //std::cout << "extended osc message\n";
            
            arg++; // message type
            
            std::string arrayTags = (arg++)->AsString();
            
            //std::cout << "arrayTags " << arrayTags << "\n";
            
            int arrayTagIndex = 0;
            
            while( arg != pReceivedMessage.ArgumentsEnd() )
            {
                if( arg->IsBool() ) oscMessage->add((arg++)->AsBoolUnchecked() );
                else if( arg->IsChar() ) oscMessage->add((arg++)->AsCharUnchecked() );
                else if( arg->IsInt32() ) oscMessage->add((arg++)->AsInt32Unchecked() );
                else if( arg->IsInt64() ) oscMessage->add((arg++)->AsInt64Unchecked() );
                else if( arg->IsFloat() ) oscMessage->add((arg++)->AsFloatUnchecked() );
                else if( arg->IsDouble() ) oscMessage->add((arg++)->AsDoubleUnchecked() );
                else if( arg->IsString() )
                {
                    std::string argString = (arg++)->AsStringUnchecked();
                    oscMessage->add(argString);
                    
                    //arrayTagIndex++;
                }
                else if( arg->IsSymbol() )
                {
                    std::string argString = (arg++)->AsSymbolUnchecked();
                    oscMessage->add(argString, OSC_TYPE_SYMBOL );
                    
                    arrayTagIndex++;
                }
                else if( arg->IsMidiMessage() ) oscMessage->add((int)(arg++)->AsMidiMessageUnchecked(), OSC_TYPE_MIDI_MESSAGE );
                else if( arg->IsTimeTag() ) oscMessage->add((long)(arg++)->AsTimeTagUnchecked(), OSC_TYPE_TIMETAG );
                else if( arg->IsBlob() )
                {
                    char * dataPtr;
                    osc::osc_bundle_element_size_t len = 0;
                    (arg++)->AsBlobUnchecked((const void*&)dataPtr, len);
                    
                    char arrayTag = arrayTags[arrayTagIndex];
                    
                    if( arrayTag == EXT_TAG_ARG_CHAR_ARRAY )
                    {
                        oscMessage->add( dataPtr, len, EXT_TYPE_ARG_CHAR_ARRAY );
                    }
                    else if( arrayTag == EXT_TAG_ARG_INT32_ARRAY )
                    {
                        int* valueData = (int*)dataPtr;
                        int valueCount = len / sizeof(int);
                        
                        oscMessage->add( valueData, valueCount, EXT_TYPE_ARG_INT32_ARRAY );
                    }
                    else if( arrayTag == EXT_TAG_ARG_INT64_ARRAY )
                    {
                        long* valueData = (long*)dataPtr;
                        int valueCount = len / sizeof(long);
                        
                        oscMessage->add( valueData, valueCount, EXT_TYPE_ARG_INT64_ARRAY );
                    }
                    else if( arrayTag == EXT_TAG_ARG_FLOAT_ARRAY )
                    {
                        float* valueData = (float*)dataPtr;
                        int valueCount = len / sizeof(float);
                        
                        oscMessage->add( valueData, valueCount, EXT_TYPE_ARG_FLOAT_ARRAY );
                    }
                    else if( arrayTag == EXT_TAG_ARG_DOUBLE_ARRAY )
                    {
                        double* valueData = (double*)dataPtr;
                        int valueCount = len / sizeof(double);
                        
                        oscMessage->add( valueData, valueCount, EXT_TYPE_ARG_DOUBLE_ARRAY );
                    }
                    
                    //delete [] dataPtr;
                    
                    arrayTagIndex++;
                }
                else if( arg->IsRgbaColor() ) oscMessage->add((int32_t)(arg++)->AsRgbaColorUnchecked(), OSC_TYPE_RGBA_COLOR );
            }
        }
        
        return oscMessage;
    }
    catch( osc::Exception& e )
    {
        // any parsing errors such as unexpected argument types, or
        // missing arguments get thrown as exceptions.
        std::stringstream ss;
        ss << "error while parsing message: " << pReceivedMessage.AddressPattern() << ": " << e.what() << "\n";
        throw Exception("OSC Error: " + ss.str() + ", failed to create message from osc::ReceivedMessage ", __FILE__, __FUNCTION__, __LINE__);
    }
    catch(Exception& e)
    {
        throw e += Exception("OSC Error: failed to create message from osc::ReceivedMessage", __FILE__, __FUNCTION__, __LINE__);
    }
}

std::shared_ptr< MultiOscMessage>
OscUtils::createMultiOscMessage( const osc::ReceivedMessage& pReceivedMessage )
{
    if( isMultiOscMessage(pReceivedMessage) == false )  throw Exception("OSC Error: ReceivedMessage does not form part of a MultiOscMessage", __FILE__, __FUNCTION__, __LINE__);
    
    try
    {
        //std::cout << "OscUtils::createMultiOscMessage begin\n";
        
        //std::cout << "message argument count " << pReceivedMessage.ArgumentCount() << "\n";
        
        std::string messageAddress = pReceivedMessage.AddressPattern();
        osc::ReceivedMessage::const_iterator arg = pReceivedMessage.ArgumentsBegin();
        
        arg++; // message type
        long messageId = (arg++)->AsInt64();
        int messageCount = (arg++)->AsInt32();
        int messageIndex = (arg++)->AsInt32();
        std::string arrayTags = (arg++)->AsString();
        
        int arrayCount = arrayTags.size();
        
        //std::cout << "arrayTags " << arrayTags << "\n";
        
        
        int arrayTagIndex = 0;
        
        std::shared_ptr< MultiOscMessage > multiOscMessage;
        
        if( isStartMultiOscMessage(pReceivedMessage) )
        {
            //std::cout << "isStartMultiOscMessage\n";
            
            //std::cout << "arrayCount " << arrayCount << "\n";
            
            Array<unsigned int> arraySizes(arrayCount);
            for(int aI=0; aI<arrayCount; ++aI)
            {
                arraySizes[aI] = (unsigned int) (arg++)->AsInt32();
            }
            
            //std::cout << "arrayCount " << arrayCount << "\n";
            
            multiOscMessage = std::shared_ptr<MultiOscMessage>( new MultiOscMessage(messageAddress, messageId, messageCount, arrayTags, arraySizes ) );
        }
        else
        {
            //std::cout << "isNextMultiOscMessage\n";
            
            multiOscMessage = std::shared_ptr<MultiOscMessage>( new MultiOscMessage(messageAddress, messageId, messageCount ) );
        }
        
        std::shared_ptr< OscMessage > oscMessage( new OscMessage(messageAddress) );
        
        int argIndex=0;
        
        while( arg != pReceivedMessage.ArgumentsEnd() )
        {
            //std::cout << "argIndex " << argIndex << "\n";
            
            if( arg->IsBool() ) oscMessage->add((arg++)->AsBoolUnchecked() );
            else if( arg->IsChar() ) oscMessage->add((arg++)->AsCharUnchecked() );
            else if( arg->IsInt32() ) oscMessage->add((arg++)->AsInt32Unchecked() );
            else if( arg->IsInt64() ) oscMessage->add((arg++)->AsInt64Unchecked() );
            else if( arg->IsFloat() ) oscMessage->add((arg++)->AsFloatUnchecked() );
            else if( arg->IsDouble() ) oscMessage->add((arg++)->AsDoubleUnchecked() );
            else if( arg->IsString() )
            {
                std::string argString = (arg++)->AsStringUnchecked();
                oscMessage->add(argString);
                
                arrayTagIndex++;
            }
            else if( arg->IsSymbol() )
            {
                std::string argString = (arg++)->AsStringUnchecked();
                oscMessage->add(argString, OSC_TYPE_SYMBOL);
                
                arrayTagIndex++;
            }
            else if( arg->IsMidiMessage() ) oscMessage->add((int)(arg++)->AsMidiMessageUnchecked(), OSC_TYPE_MIDI_MESSAGE );
            else if( arg->IsTimeTag() ) oscMessage->add((long)(arg++)->AsTimeTagUnchecked(), OSC_TYPE_TIMETAG );
            else if( arg->IsBlob() )
            {
                char * dataPtr;
                osc::osc_bundle_element_size_t len = 0;
                (arg++)->AsBlobUnchecked((const void*&)dataPtr, len);
                
                char arrayTag = arrayTags[arrayTagIndex];
                
                if( arrayTag == EXT_TAG_ARG_CHAR_ARRAY )
                {
                    oscMessage->add( dataPtr, len, EXT_TYPE_ARG_CHAR_ARRAY );
                }
                else if( arrayTag == EXT_TAG_ARG_INT32_ARRAY )
                {
                    int* valueData = (int*)dataPtr;
                    int valueCount = len / sizeof(int);
                    
                    oscMessage->add( valueData, valueCount, EXT_TYPE_ARG_INT32_ARRAY );
                }
                else if( arrayTag == EXT_TAG_ARG_INT64_ARRAY )
                {
                    long* valueData = (long*)dataPtr;
                    int valueCount = len / sizeof(long);
                    
                    oscMessage->add( valueData, valueCount, EXT_TYPE_ARG_INT64_ARRAY );
                }
                else if( arrayTag == EXT_TAG_ARG_FLOAT_ARRAY )
                {
                    float* valueData = (float*)dataPtr;
                    int valueCount = len / sizeof(float);
                    
                    oscMessage->add( valueData, valueCount, EXT_TYPE_ARG_FLOAT_ARRAY );
                }
                else if( arrayTag == EXT_TAG_ARG_DOUBLE_ARRAY )
                {
                    double* valueData = (double*)dataPtr;
                    int valueCount = len / sizeof(double);
                    
                    oscMessage->add( valueData, valueCount, EXT_TYPE_ARG_DOUBLE_ARRAY );
                }
                
                //delete [] dataPtr;
                
                arrayTagIndex++;
            }
            else if( arg->IsRgbaColor() ) oscMessage->add((int)(arg++)->AsRgbaColorUnchecked(), OSC_TYPE_RGBA_COLOR );
            
            argIndex++;
        }

        multiOscMessage->setPartialMessage(messageIndex, oscMessage );
        
        //std::cout << "OscUtils::createMultiOscMessage end\n";
        
        return multiOscMessage;
        
    }
    catch( osc::Exception& e )
    {
        // any parsing errors such as unexpected argument types, or
        // missing arguments get thrown as exceptions.
        std::stringstream ss;
        ss << "error while parsing message: " << pReceivedMessage.AddressPattern() << ": " << e.what() << "\n";
        throw Exception("OSC Error: " + ss.str() + ", failed to create message from osc::ReceivedMessage ", __FILE__, __FUNCTION__, __LINE__);
    }
    catch(Exception& e)
    {
        throw e += Exception("OSC Error: failed to create MultiOscMessage from osc::ReceivedMessage", __FILE__, __FUNCTION__, __LINE__);
    }
}

void
OscUtils::updateMultiOscMessage( const osc::ReceivedMessage& pReceivedMessage, std::shared_ptr< MultiOscMessage> pMultiOscMessage )
{
    if( isMultiOscMessage(pReceivedMessage) == false )  throw Exception("OSC Error: ReceivedMessage does not form part of a MultiOscMessage", __FILE__, __FUNCTION__, __LINE__);
    
    try
    {
        //std::cout << "OscUtils::updateMultiOscMessage begin\n";
        
        //std::cout << "message argument count " << pReceivedMessage.ArgumentCount() << "\n";
        
        std::string multiOscMessageAddress = pMultiOscMessage->address();
        unsigned long multiOscMessageId = pMultiOscMessage->messageId();
        
        std::string messageAddress = pReceivedMessage.AddressPattern();
        
        if( multiOscMessageAddress.compare(messageAddress ) != 0 ) throw Exception("OSC Error: address of ReceivedMessage " + messageAddress + " does not match address of MultiOscMessage" + multiOscMessageAddress, __FILE__, __FUNCTION__, __LINE__);
        
        osc::ReceivedMessage::const_iterator arg = pReceivedMessage.ArgumentsBegin();
        
        arg++; // message type
        long messageId = (arg++)->AsInt64();
        
        if( messageId != multiOscMessageId ) throw Exception("OSC Error: messageId of ReceivedMessage " + std::to_string(messageId) + " does not match messageId of MultiOscMessage" + std::to_string(multiOscMessageId), __FILE__, __FUNCTION__, __LINE__);
        
        int messageCount = (arg++)->AsInt32();
        int messageIndex = (arg++)->AsInt32();
        
        //std::cout << "message count " << messageCount << "\n";
        //std::cout << "message index " << messageIndex << "\n";
        
        std::string arrayTags = (arg++)->AsString();
        
        //std::cout << "arrayTags " << arrayTags << "\n";
        
        int arrayCount = arrayTags.size();
        int arrayTagIndex = 0;
        
        //std::cout << "array count " << arrayCount << "\n";
        
        if( isStartMultiOscMessage(pReceivedMessage) )
        {
            //std::cout << "isStartMultiOscMessage\n";
            
            Array<unsigned int> arraySizes(arrayCount);
            for(int aI=0; aI<arrayCount; ++aI)
            {
                arraySizes[aI] = (unsigned int) (arg++)->AsInt32();
            }
            
            pMultiOscMessage->setArrayTags(arrayTags);
            pMultiOscMessage->setArraySizes(arraySizes);
        }
        else
        {
            //std::cout << "isNextMultiOscMessage\n";
        }
        
        std::shared_ptr< OscMessage > oscMessage( new OscMessage(messageAddress) );
        
        while( arg != pReceivedMessage.ArgumentsEnd() )
        {
            if( arg->IsBool() ) oscMessage->add((arg++)->AsBoolUnchecked() );
            else if( arg->IsChar() ) oscMessage->add((arg++)->AsCharUnchecked() );
            else if( arg->IsInt32() ) oscMessage->add((arg++)->AsInt32Unchecked() );
            else if( arg->IsInt64() ) oscMessage->add((arg++)->AsInt64Unchecked() );
            else if( arg->IsFloat() ) oscMessage->add((arg++)->AsFloatUnchecked() );
            else if( arg->IsDouble() ) oscMessage->add((arg++)->AsDoubleUnchecked() );
            else if( arg->IsString() )
            {
                std::string argString = (arg++)->AsStringUnchecked();
                oscMessage->add(argString);
                
                arrayTagIndex++;
            }
            else if( arg->IsSymbol() )
            {
                std::string argString = (arg++)->AsStringUnchecked();
                oscMessage->add(argString, OSC_TYPE_SYMBOL);
                
                arrayTagIndex++;
            }
            else if( arg->IsMidiMessage() ) oscMessage->add((int)(arg++)->AsMidiMessageUnchecked(), OSC_TYPE_MIDI_MESSAGE );
            else if( arg->IsTimeTag() ) oscMessage->add((long)(arg++)->AsTimeTagUnchecked(), OSC_TYPE_TIMETAG );
            else if( arg->IsBlob() )
            {
                //std::cout << "is array\n";
                
                char * dataPtr;
                osc::osc_bundle_element_size_t len = 0;
                (arg++)->AsBlobUnchecked((const void*&)dataPtr, len);
                
                char arrayTag = arrayTags[arrayTagIndex];
                
                //std::cout << "array tag index " << arrayTagIndex << "\n";
                //std::cout << "array tag " << arrayTag << "\n";
                
                if( arrayTag == EXT_TAG_ARG_CHAR_ARRAY )
                {
                    oscMessage->add( dataPtr, len, EXT_TYPE_ARG_CHAR_ARRAY );
                }
                else if( arrayTag == EXT_TAG_ARG_INT32_ARRAY )
                {
                    int* valueData = (int*)dataPtr;
                    int valueCount = len / sizeof(int);
                    
                    oscMessage->add( valueData, valueCount, EXT_TYPE_ARG_INT32_ARRAY );
                }
                else if( arrayTag == EXT_TAG_ARG_INT64_ARRAY )
                {
                    long* valueData = (long*)dataPtr;
                    int valueCount = len / sizeof(long);
                    
                    oscMessage->add( valueData, valueCount, EXT_TYPE_ARG_INT64_ARRAY );
                }
                else if( arrayTag == EXT_TAG_ARG_FLOAT_ARRAY )
                {
                    float* valueData = (float*)dataPtr;
                    int valueCount = len / sizeof(float);
                    
                    oscMessage->add( valueData, valueCount, EXT_TYPE_ARG_FLOAT_ARRAY );
                }
                else if( arrayTag == EXT_TAG_ARG_DOUBLE_ARRAY )
                {
                    double* valueData = (double*)dataPtr;
                    int valueCount = len / sizeof(double);
                    
                    oscMessage->add( valueData, valueCount, EXT_TYPE_ARG_DOUBLE_ARRAY );
                }
                
                //delete [] dataPtr;
                
                arrayTagIndex++;
            }
            else if( arg->IsRgbaColor() ) oscMessage->add((int)(arg++)->AsRgbaColorUnchecked(), OSC_TYPE_RGBA_COLOR );
        }
        
        pMultiOscMessage->setPartialMessage(messageIndex, oscMessage );
        
        //std::cout << "OscUtils::updateMultiOscMessage end\n";
        
    }
    catch( osc::Exception& e )
    {
        // any parsing errors such as unexpected argument types, or
        // missing arguments get thrown as exceptions.
        std::stringstream ss;
        ss << "error while parsing message: " << pReceivedMessage.AddressPattern() << ": " << e.what() << "\n";
        throw Exception("OSC Error: " + ss.str() + ", failed to create message from osc::ReceivedMessage ", __FILE__, __FUNCTION__, __LINE__);
    }
    catch(Exception& e)
    {
        throw e += Exception("OSC Error: failed to create MultiOscMessage from osc::ReceivedMessage", __FILE__, __FUNCTION__, __LINE__);
    }
}

std::shared_ptr<osc::OutboundPacketStream>
OscUtils::createPacketStream( std::shared_ptr<OscMessage> pMessage )
{
    try
    {
        char buffer[sMaxOscMessageSize];
        std::shared_ptr<osc::OutboundPacketStream> packetStream( new osc::OutboundPacketStream( buffer, sMaxOscMessageSize ) );
        
        unsigned int arrayCount = pMessage->arrayCount();
        unsigned int messagePacketSize = extendedMessagePacketSize(pMessage);
        
        if( messagePacketSize < sMaxOscMessageSize && arrayCount == 0 ) // create a conventional message
        {
            (*packetStream) << osc::BeginMessage( pMessage->address().c_str() );
            
            std::vector< _OscArg* > arguments = pMessage->arguments();
            unsigned int argumentCount = arguments.size();
            
            for(int aI=0; aI<argumentCount; ++aI)
            {
                arguments[aI]->addToPacketStream(*packetStream);
            }
        
            (*packetStream) << osc::EndMessage;
        }
        else if( messagePacketSize < sMaxOscMessageSize && arrayCount > 0  ) // create extended message
        {
            (*packetStream) << osc::BeginMessage( pMessage->address().c_str() );
            (*packetStream) << 'e';
            
            const std::string& arrayTags = pMessage->arrayTags();
            
            (*packetStream) << arrayTags.c_str();
            
            std::vector< _OscArg* > arguments = pMessage->arguments();
            unsigned int argumentCount = arguments.size();
            
            for(int aI=0; aI<argumentCount; ++aI)
            {
                arguments[aI]->addToPacketStream(*packetStream);
            }
            
            (*packetStream) << osc::EndMessage;
        }
        
        return packetStream;
    }
    catch( Exception& e)
    {
        throw e += Exception("OSC Error: Failed to create packet streams", __FILE__, __FUNCTION__, __LINE__);
    }
}

std::vector< std::shared_ptr<osc::OutboundPacketStream> >
OscUtils::createPacketStreams( std::shared_ptr<MultiOscMessage> pMultiOscMessage )
{
    try
    {
        if( pMultiOscMessage->complete() == false ) throw ("OSC Error: multi osc message is not complete, creating packet streams impossible", __FILE__, __FUNCTION__, __LINE__);
        
        const std::vector< std::shared_ptr<OscMessage> >& partialMessages = pMultiOscMessage->partialMessages();
        unsigned int partialMessageCount = partialMessages.size();
        
        if( partialMessageCount == 0 ) throw ("OSC Error: multi osc message contains no partial messages, creating packet streams impossible", __FILE__, __FUNCTION__, __LINE__);
        
        const std::string& messageAddress = pMultiOscMessage->address();
        int64_t messageId = pMultiOscMessage->messageId();
        const Array<unsigned int >& arraySizes = pMultiOscMessage->arraySizes();
        unsigned int arrayCount = arraySizes.size();
        std::string arrayTags = pMultiOscMessage->arrayTags();
        
        //std::cout << "arrayTags " << arrayTags << " arraySizes " << arraySizes << "\n";
        
        std::vector< _OscArg* > arguments;
        unsigned int argumentCount;
        
        std::vector< std::shared_ptr<osc::OutboundPacketStream> > packetStreams;
        std::shared_ptr<osc::OutboundPacketStream> packetStream;
        
         // first partial message
        auto bufferDeleter = [](osc::OutboundPacketStream* pStream)
        {
            //std::cout << "[deleter called]\n";
            delete pStream->Data();
            delete pStream;
        };
		packetStream = std::shared_ptr<osc::OutboundPacketStream>(new osc::OutboundPacketStream(new char[sMaxOscMessageSize], sMaxOscMessageSize), bufferDeleter);

        (*packetStream) << osc::BeginMessage( messageAddress.c_str() );
        (*packetStream) << 'm';
        (*packetStream) << (osc::int64)messageId;
        (*packetStream) << (osc::int32)partialMessageCount;
        (*packetStream) << (osc::int32)0;
        (*packetStream) << arrayTags.c_str();
        
        //std::cout << "arrayTags " << arrayTags << "\n";
        //std::cout << "arrayCount " << arrayCount << "\n";
        
        for(int aI=0; aI<arrayCount; ++aI)
        {
            (*packetStream) << (int)arraySizes[aI];
            
            //std::cout << "aI " << aI << " size " << (int)arraySizes[aI] << "\n";
        }
        
        arguments = partialMessages[0]->arguments();
        argumentCount = arguments.size();
        
        //std::cout << "start message\n";
        
        for(int aI=0; aI<argumentCount; ++aI)
        {
            //std::cout << "argument " << aI << " valueCount " << arguments[aI]->valueCount() << "\n";
            
            arguments[aI]->addToPacketStream(*packetStream);
        }
        
        (*packetStream) << osc::EndMessage;
        
        packetStreams.push_back(packetStream);
        
        // next partial messages
        for(unsigned int partialMessageIndex=1; partialMessageIndex < partialMessageCount; ++partialMessageIndex)
        {
            //std::cout << "partialMessageIndex " << partialMessageIndex << "\n";
   
            auto bufferDeleter = [](osc::OutboundPacketStream* pStream)
            {
                //std::cout << "[deleter called]\n";
                delete pStream->Data();
                delete pStream;
            };
 			packetStream = std::shared_ptr<osc::OutboundPacketStream>(new osc::OutboundPacketStream(new char[sMaxOscMessageSize], sMaxOscMessageSize), bufferDeleter);


            (*packetStream) << osc::BeginMessage( messageAddress.c_str() );
            (*packetStream) << (char)'m';
            (*packetStream) << (osc::int64)messageId;
            (*packetStream) << (int)partialMessageCount;
            (*packetStream) << (int)partialMessageIndex;

            arguments = partialMessages[partialMessageIndex]->arguments();
            argumentCount = arguments.size();
            arrayTags = partialMessages[partialMessageIndex]->arrayTags();
            
            (*packetStream) << arrayTags.c_str();
            
            for(int aI=0; aI<argumentCount; ++aI)
            {
                //std::cout << "argument " << aI << "type " << typeName(arguments[aI]->oscType()) << " valueCount " << arguments[aI]->valueCount() << "\n";
                
                 arguments[aI]->addToPacketStream(*packetStream);
            }
            
            (*packetStream) << osc::EndMessage;
            
            packetStreams.push_back(packetStream);
        }
        
        return packetStreams;
    }
    catch (Exception& e)
    {
        throw e += Exception( "OSC Error: failed to create packet streams", __FILE__, __FUNCTION__, __LINE__ );
    }
   
}

std::shared_ptr<MultiOscMessage>
OscUtils::splitMessage( std::shared_ptr<OscMessage> pMessage )
{
    try
    {
        //std::cout << "OscUtils::splitMessage begin\n";
        
        std::string messageAddress = pMessage->address();
        int messageId = ofGetSystemTimeMicros();
        const std::vector< _OscArg* >& messageArguments = pMessage->arguments();
        unsigned int argumentCount = messageArguments.size();
        Array<unsigned int> arraySizes = pMessage->arraySizes();
        std::string arrayTags = pMessage->arrayTags();
        
        // create partial messages
        int remainingMessagePacketSize;
        std::vector< std::shared_ptr< OscMessage> > partialMessages;
        std::shared_ptr<OscMessage> partialMessage;
        int argumentIndex = 0;
        int arrayValueIndex = 0;
        int messageCount;
        
        // >* address (string)
        // >* 0: message type (char) , always 'm' at the moment
        // >* 1: message id (int64)
        // >* 2: message count (int32) > 1
        // >* 3: message index (int32) = 0
        // >* 4: message array tags (string)
        // >* 5: message full array arg sizes, size0 ... sizeN (int32)
        // *>=5+: message arguments
        
        // start message
        messageCount = 1;
        
        partialMessage = std::shared_ptr<OscMessage>( new OscMessage(messageAddress) );
        partialMessages.push_back(partialMessage);
        
        remainingMessagePacketSize = sMaxOscMessageSize;
        remainingMessagePacketSize -= sEmptyStartMultiMessageHeaderSize; // default empty message header
        remainingMessagePacketSize -= (8 + (messageAddress.size() - 1 ) / 4 * 4); // message address
        remainingMessagePacketSize -= arraySizes.size() * sizeof(char); // array type tags
        remainingMessagePacketSize -= arraySizes.size() * sizeof(int); // array sizes
        
        if( remainingMessagePacketSize <= 0 ) throw Exception( "OSC Error: the header of the start multi message " + messageAddress + " is too big to fit into a UDP packet ", __FILE__, __FUNCTION__, __LINE__ );
        
        while( argumentIndex < argumentCount )
        {
            const _OscArg* oscArg = messageArguments[argumentIndex];
            
            bool isArrayType = oscArg->valueCount() > 1;
            
            OscType oscType = oscArg->oscType();
            int argTypeSize = oscArg->typeSize();
            int argByteCount = oscArg->byteCount();
            
            //std::cout << "argIndex " << argumentIndex << " oscType " << sOscTypeNames[oscType] << " valueCount " << oscArg->valueCount() << " byteCount " << argByteCount << " messageCount " << messageCount << "\n";
            
            if( isArrayType == true )
            {
                //std::cout << "array arg\n";
                //std::cout << "arrayValueIndex " << arrayValueIndex << "\n";
                
                int arrayPacketSize;
                int arrayValueCount = oscArg->valueCount();
                int arrayHeaderSize;
                
                if( messageCount == 1 ) arrayHeaderSize = sStartMultiMessageArrayHeaderSize;
                else arrayHeaderSize = sMultiMessageArrayHeaderSize;
                
                //std::cout << "arrayHeaderSize " << arrayHeaderSize << "\n";
                
                if( arrayValueIndex == 0 ) // this is a new array
                {
                    arrayPacketSize = arrayHeaderSize + argByteCount;
                    
                    //std::cout << "array " << arrayValueIndex << " valueCount " << oscArg->valueCount() << " byteCount " << argByteCount << "\n";
                    
                    //std::cout << "arrayPacketSize " << arrayPacketSize << " remainingMessagePacketSize " << remainingMessagePacketSize << "\n";
                    
                    if( arrayPacketSize < remainingMessagePacketSize ) // we can fit the entire array into this udp packet
                    {
                        //std::cout << "full array fit\n";
                        
                        partialMessage->mArguments.push_back( oscArg->copy() );
                        remainingMessagePacketSize -= arrayPacketSize;
                        argumentIndex++;
                    }
                    else // we can only fit the first part of the array into this udp packet
                    {
                        //std::cout << "first array part fit\n";
                        
                        arrayPacketSize = remainingMessagePacketSize;
                        
                        int storeableValueCount = std::max( (arrayPacketSize - arrayHeaderSize) / argTypeSize - 2, 0 );
                        
                        //std::cout << "arrayPacketSize " << arrayPacketSize << " storeableValueCount " << storeableValueCount << "\n";
                        
                        if( storeableValueCount > 0 )
                        {
                            _OscArg* partialArrayArg = oscArg->copy(0, storeableValueCount);
                            
                            //std::cout << "partial array valueCount " << partialArrayArg->valueCount() << " byteCount " << partialArrayArg->byteCount() << "\n";
                            
                            partialMessage->mArguments.push_back( partialArrayArg );
                            
                            arrayValueIndex = storeableValueCount;
                        }
                        
                        messageCount += 1;
                    }
                }
                else // are already added a part of this array to the previous osc message
                {
                    arrayPacketSize = arrayHeaderSize + argByteCount - arrayValueIndex * argTypeSize;
                    
                    //std::cout << "arrayHeaderSize " << arrayHeaderSize << " argByteCount " << argByteCount << " arrayValueIndex " << arrayValueIndex << " argTypeSize " << argTypeSize << "\n";
                    
                    //std::cout << " arrayPacketSize " << arrayPacketSize << " remainingMessagePacketSize " << remainingMessagePacketSize << "\n";
                    
                    if( arrayPacketSize < remainingMessagePacketSize ) // we can fit the entire remaining part of the array into this udp packet
                    {
                        //std::cout << "remining array part fit\n";
                        
                        _OscArg* partialArrayArg = oscArg->copy(arrayValueIndex, arrayValueCount - arrayValueIndex );
                        
                        //std::cout << "partial array valueCount " << partialArrayArg->valueCount() << " byteCount " << partialArrayArg->byteCount() << "\n";
                        
                        partialMessage->mArguments.push_back( partialArrayArg );
                        
                        remainingMessagePacketSize -= arrayPacketSize;
                        arrayValueIndex = 0;
                        argumentIndex++;
                    }
                    else // we can only fit part of the array into this udp packet
                    {
                        //std::cout << "next array part fit\n";
                        
                        arrayPacketSize = remainingMessagePacketSize;
                        
                        int storeableValueCount = std::max( (arrayPacketSize - arrayHeaderSize) / argTypeSize - 2, 0 );
                        
                        //std::cout << "storeableValueCount " << storeableValueCount << "\n";
                        
                        if( storeableValueCount > 0 )
                        {
                            _OscArg* partialArrayArg = oscArg->copy(arrayValueIndex, storeableValueCount);
                            
                            //std::cout << "partial array valueCount " << partialArrayArg->valueCount() << " byteCount " << partialArrayArg->byteCount() << "\n";
                            
                            partialMessage->mArguments.push_back( partialArrayArg );
                            
                            arrayValueIndex += storeableValueCount;
                        }
                        
                        messageCount += 1;
                    }
                }
            }
            else
            {
                //std::cout << "non array arg\n";
                
                int argPacketSize;
                
                if( messageCount == 1 ) argPacketSize = startMultiMessagePacketSize(*oscArg);
                else argPacketSize = multiMessagePacketSize(*oscArg);
                
                //std::cout << "argPacketSize " << argPacketSize << " remainingMessagePacketSize " << remainingMessagePacketSize << "\n";
                
                if( argPacketSize < remainingMessagePacketSize ) // we can fit the argument into this udp packet
                {
                    //std::cout << "fit\n";
                    
                    partialMessage->mArguments.push_back( oscArg->copy() );
                    remainingMessagePacketSize -= argPacketSize;
                    argumentIndex++;
                }
                else // argument doesn't fit into this packet
                {
                    //std::cout << "non fit\n";
                    
                    messageCount += 1;
                }
            }
            
            if( messageCount > partialMessages.size() && argumentIndex < argumentCount )
            {
                //std::cout << "messageCount " << messageCount << " partialMessages.size() " << partialMessages.size() << " argumentIndex " << argumentIndex << " argumentCount " << argumentCount  << "\n";
                
                partialMessage = std::shared_ptr<OscMessage>( new OscMessage(messageAddress) );
                partialMessages.push_back(partialMessage);
                
                remainingMessagePacketSize = sMaxOscMessageSize;
                remainingMessagePacketSize -= sEmptyMultiMessageHeaderSize; // default empty message header
                remainingMessagePacketSize -= (8 + (messageAddress.size() - 1 ) / 4 * 4); // message address
            }
        }
        
        std::shared_ptr<MultiOscMessage> multiOscMessage( new MultiOscMessage( partialMessages, messageId, arrayTags, arraySizes ) );
        
//        //debug
//        for(int i=0; i<partialMessages.size(); ++i)
//        {
//            std::cout << "partialMessage " << i << "\n";
//            std::cout << *partialMessages[i] << "\n";
//        }
//        //debug done
        
        return multiOscMessage;
    }
    catch (Exception& e)
    {
        throw e += Exception("Osc Error: failed to split osc messages", __FILE__, __FUNCTION__, __LINE__);
    }
    
    //std::cout << "OscUtils::splitMessage end\n";
}

std::shared_ptr<OscMessage>
OscUtils::mergeMessage( std::shared_ptr<MultiOscMessage> pMultiMessage )
{
    //std::cout << "OscUtils::mergeMessage begin\n";
    
    try
    {
        if( pMultiMessage->complete() == false ) throw ("OSC Error: multi osc message is not complete, merging messages impossible", __FILE__, __FUNCTION__, __LINE__);
        
        const std::string& messageAddres = pMultiMessage->mMessageAddress;
        const Array< unsigned int >& arraySizes = pMultiMessage->mArraySizes;
        const std::string& arrayTags = pMultiMessage->mArrayTags;
        
        //std::cout << "arraySizes " << arraySizes << "\n";
        
        //std::cout << "OscUtils::mergeMessage( std::shared_ptr<MultiOscMessage> pMultiMessage )\n";
        //std::cout << "this " << pMultiMessage << " arrayTags " << arrayTags << "\n";
        
        const std::vector< std::shared_ptr<OscMessage> >& partialMessages = pMultiMessage->mPartialMessages;
        
        if( partialMessages.size() == 0 ) throw Exception("OSC Error: multi osc message doesn't contain any partial messages\n", __FILE__, __FUNCTION__, __LINE__);
        
        std::shared_ptr< OscMessage > oscMessage( new OscMessage( messageAddres ) );
        pMultiMessage->mMessage = oscMessage;
        
        int arraySizeIndex = 0;
        int arrayValueIndex = 0;
        _OscArg* arrayArg = NULL;
        
        unsigned int partialMessageCount = partialMessages.size();
        
        for(int partialMessageIndex=0; partialMessageIndex < partialMessageCount; ++partialMessageIndex )
        {
            //std::cout << "partialMessageIndex " << partialMessageIndex << "\n";
            
            std::shared_ptr<OscMessage> partialMessage = partialMessages[partialMessageIndex];
            
            const std::vector< _OscArg* >& partialArguments = partialMessage->arguments();
            unsigned int partialArgumentCount = partialArguments.size();
            
            for(int partialArgumentIndex=0; partialArgumentIndex<partialArgumentCount; ++partialArgumentIndex)
            {
                _OscArg* partialArgument = partialArguments[partialArgumentIndex];
                
                //std::cout << "message index " << partialMessageIndex << " arg index " << partialArgumentIndex << " arg type " << typeName(partialArgument->oscType()) << " valueCount " << partialArgument->valueCount() << "\n";
                
                bool isArrayType = partialArgument->valueCount() > 1;
                
                if( isArrayType == true )
                {
                    if( arrayValueIndex == 0 ) // new array
                    {
                        //std::cout << "new array\n";
                        
                        int partialArrayValueCount = partialArgument->valueCount();

                        if( partialArrayValueCount == arraySizes[arraySizeIndex] ) // the whole array is stored in this argument
                        {
                            //std::cout << "full array\n";
                            
                            oscMessage->mArguments.push_back( partialArgument->copy() );
                            
                            arraySizeIndex++;
                        }
                        else // only the first part of the array is stored in this argument
                        {
                            //std::cout << "first part of new array\n";
                            
                            arrayArg = partialArgument->copy();
                            arrayValueIndex += partialArrayValueCount;
                        }
                    }
                    else // we are in the process of expanding an existing array
                    {
                        int partialArrayValueCount = partialArgument->valueCount();
                        
                        //std::cout << "arrayValueIndex " << arrayValueIndex << " partialArrayValueCount " << partialArrayValueCount << " fullArraySize " << arraySizes[arraySizeIndex] << " currentArraySize " << ( arrayValueIndex + partialArrayValueCount ) << "\n";
                        
                        if( arrayValueIndex + partialArrayValueCount == arraySizes[arraySizeIndex] ) // this part of the array completes the full array
                        {
                            //std::cout << "complete array\n";
                            
                            arrayArg->append(*partialArgument);
                            oscMessage->mArguments.push_back( arrayArg );
                            arrayArg = NULL;
                            arrayValueIndex = 0;
                            arraySizeIndex++;
                        }
                        else // this part of the array doesn't complete full array
                        {
                            //std::cout << "expand array\n";
                            
                            arrayArg->append(*partialArgument);
                            arrayValueIndex += partialArrayValueCount;
                        }
                    }
                }
                else
                {
                    //std::cout << "not an array\n";
                    
                    oscMessage->mArguments.push_back( partialArgument->copy() );
                }
            }
        }
        
        return oscMessage;
    }
    catch (Exception& e)
    {
        throw e += Exception("Osc Error: failed to merge os messages", __FILE__, __FUNCTION__, __LINE__);
    }
}


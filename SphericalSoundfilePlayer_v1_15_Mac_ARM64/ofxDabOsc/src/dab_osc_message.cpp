/** \file dab_osc_message.cpp
 */

#include "dab_osc_message.h"
#include "dab_osc_arg.h"
#include "ofUtils.h"

#include <sstream>

using namespace dab;

#pragma mark OscMessage implementation

OscMessage::OscMessage()
{}

OscMessage::OscMessage( const std::string& pAddress )
: mAddress(pAddress)
{}

OscMessage::OscMessage( const OscMessage& pOscMessage )
: mAddress(pOscMessage.mAddress)
{
    int argCount = pOscMessage.mArguments.size();
    mArguments.resize(argCount);
    
    for(int aI=0; aI<argCount; ++aI) mArguments[aI] = pOscMessage.mArguments[aI]->copy();
}

OscMessage::~OscMessage()
{
    clear();
}

const std::string&
OscMessage::address() const
{
    return mAddress;
}

unsigned int
OscMessage::argumentCount() const
{
    return mArguments.size();
}

std::string
OscMessage::argumentTags() const
{
    OscUtils& oscUtils = OscUtils::get();
    
    std::string argTags;
    
    int argumentCount = mArguments.size();
    
    for(int aI=0; aI<argumentCount; ++aI)
    {
        argTags += oscUtils.oscTag(mArguments[aI]->oscType());
    }
    
    return argTags;
}

unsigned int
OscMessage::arrayCount() const
{
    int _arrayCount = 0;
    
    int argumentCount = mArguments.size();
    
    for(int aI=0; aI<argumentCount; ++aI)
    {
        if( mArguments[aI]->valueCount() > 1 && mArguments[aI]->oscType() != OSC_TYPE_STRING) _arrayCount++;
    }
    
    return _arrayCount;
}

std::string
OscMessage::arrayTags() const
{
    OscUtils& oscUtils = OscUtils::get();
    
    std::string _arrayTags;
    
    int argumentCount = mArguments.size();
    
    for(int aI=0; aI<argumentCount; ++aI)
    {
        if( mArguments[aI]->valueCount() > 1 && mArguments[aI]->oscType() != OSC_TYPE_STRING ) _arrayTags += oscUtils.oscArrayTag(mArguments[aI]->oscType());
    }
    
    return _arrayTags;
}

Array<unsigned int>
OscMessage::arraySizes() const
{
    std::vector<unsigned int> _arraySizes;
    
    int argumentCount = mArguments.size();
    
    for(int aI=0; aI<argumentCount; ++aI)
    {
        if( mArguments[aI]->valueCount() > 1 ) _arraySizes.push_back( mArguments[aI]->valueCount() );
    }
    
    return Array<unsigned int>( _arraySizes.size(), _arraySizes.data() );
}

bool
OscMessage::argumentMatch( unsigned int pArgIndex, OscType pOscType ) const
{
    if( pArgIndex >= mArguments.size() ) throw ("OSC Error: requested argument " + std::to_string(pArgIndex) + " exceeds number of arguments available " + std::to_string( mArguments.size() ), __FILE__, __FUNCTION__, __LINE__  );
    
    return pOscType == mArguments[pArgIndex]->oscType();
}

bool
OscMessage::argumentMatch( const std::initializer_list<OscType>& pOscTypes ) const
{
    if( pOscTypes.size() > mArguments.size() ) throw ("OSC Error: number of osc types requested " + std::to_string(pOscTypes.size()) + " exceeds number of arguments available " + std::to_string( mArguments.size() ), __FILE__, __FUNCTION__, __LINE__  );
    
    int argMatchCount = pOscTypes.size();
    
    int aI=0;
    for(auto iter = pOscTypes.begin(); iter != pOscTypes.end(); ++iter, ++aI)
    {
        if( *iter != mArguments[aI]->oscType() ) return false;
    }
    
    return true;
}

unsigned int
OscMessage::valueCount(unsigned int pArgIndex) const
{
    if( pArgIndex >= mArguments.size() ) throw ("OSC Error: requested argument " + std::to_string(pArgIndex) + " exceeds number of arguments available " + std::to_string( mArguments.size() ), __FILE__, __FUNCTION__, __LINE__  );
    
    return mArguments[pArgIndex]->valueCount();
}

const _OscArg*
OscMessage::operator[]( unsigned int pArgIndex ) const
{
    if( pArgIndex >= mArguments.size() ) throw ("OSC Error: requested argument " + std::to_string(pArgIndex) + " exceeds number of arguments available " + std::to_string( mArguments.size() ), __FILE__, __FUNCTION__, __LINE__  );
    
    return mArguments[pArgIndex];
}

const std::vector< _OscArg* >&
OscMessage::arguments() const
{
    return mArguments;
}

void
OscMessage::clear()
{
    int argCount = mArguments.size();
    for(int aI=0; aI<argCount; ++aI) delete mArguments[aI];
    mArguments.clear();
}

void
OscMessage::setAddress( const std::string& pAddress )
{
    mAddress = pAddress;
}

unsigned int
OscMessage::byteSize() const
{
    // TODO
	return 0;
}

OscMessage::operator std::string() const
{
    std::stringstream ss;
    
    ss << "address: " << mAddress << "\n";
    
    int argumentCount = mArguments.size();
    
    for(int aI=0; aI<argumentCount; ++aI) ss << "argument " << aI << " :\n" << *(mArguments[aI]);
    
    return ss.str();
}

#pragma mark MultiOscMessage implementation


MultiOscMessage::MultiOscMessage()
: mTotalPartialMessageCount(0)
, mCurrentPartialMessageCount(0)
, mComplete(false)
{}

MultiOscMessage::MultiOscMessage( const std::string& pAddress, unsigned long pId, unsigned int pPartialMessageCount )
: mMessageAddress(pAddress)
, mMessageId(pId)
, mTotalPartialMessageCount(pPartialMessageCount)
, mCurrentPartialMessageCount(0)
, mComplete(false)
{
    mPartialMessages.resize(mTotalPartialMessageCount);
}

MultiOscMessage::MultiOscMessage( const std::string& pAddress, unsigned long pId, unsigned int pPartialMessageCount, const std::string& pArrayTags, const Array<unsigned int>& pArraySizes )
: mMessageAddress(pAddress)
, mMessageId(pId)
, mArrayTags(pArrayTags)
, mArraySizes(pArraySizes)
, mTotalPartialMessageCount(pPartialMessageCount)
, mCurrentPartialMessageCount(0)
, mComplete(false)
{
    mPartialMessages.resize(mTotalPartialMessageCount);
}

MultiOscMessage::MultiOscMessage( const std::vector< std::shared_ptr< OscMessage> >& pPartialMessages, unsigned long pId, const std::string& pArrayTags, const Array<unsigned int>& pArraySizes )
: mPartialMessages(pPartialMessages)
, mMessageId(pId)
, mArrayTags(pArrayTags)
, mArraySizes(pArraySizes)
, mTotalPartialMessageCount(mPartialMessages.size())
, mCurrentPartialMessageCount(mTotalPartialMessageCount)
, mComplete(true)
{
    mMessageAddress = mPartialMessages[0]->address();
}

MultiOscMessage::MultiOscMessage( std::shared_ptr<OscMessage> pOscMessage, unsigned long pId )
: mMessageAddress(pOscMessage->address())
, mMessageId(pId)
, mMessage(pOscMessage)
, mTotalPartialMessageCount(0)
, mCurrentPartialMessageCount(0)
, mComplete(true)
{}

MultiOscMessage::~MultiOscMessage()
{
    mPartialMessages.clear();
}

const std::string&
MultiOscMessage::address() const
{
    return mMessageAddress;
}

unsigned long
MultiOscMessage::messageId() const
{
    return mMessageId;
}

const std::string&
MultiOscMessage::arrayTags() const
{
    return mArrayTags;
}

const Array< unsigned int >&
MultiOscMessage::arraySizes() const
{
    return mArraySizes;
}

bool
MultiOscMessage::complete() const
{
    return mComplete;
}

std::shared_ptr<OscMessage>
MultiOscMessage::message() const
{
    if( mComplete == false ) throw Exception("OSC Error: multi message doesn't contain a complete osc message", __FILE__, __FUNCTION__, __LINE__);
    
    return mMessage;
}

const std::vector< std::shared_ptr<OscMessage> >&
MultiOscMessage::partialMessages() const
{
    return mPartialMessages;
}

void
MultiOscMessage::setArrayTags( const std::string& pArrayTags )
{
    mArrayTags = pArrayTags;
    
//    std::cout << "setArrayTags( const std::string& pArrayTags )\n";
//    std::cout << "this " << this << " arrayTags " <<  mArrayTags << "\n";
}

void
MultiOscMessage::setArraySizes(const Array<unsigned int> &pArraySizes)
{
    mArraySizes = pArraySizes;
}

void
MultiOscMessage::setPartialMessage( unsigned int pIndex, std::shared_ptr<OscMessage> pPartialMessage )
{
    //std::cout << "setPartialMessage index " << pIndex << "\n";
    
    int partialMessageCount = mPartialMessages.size();
//    for(int pI=0; pI<partialMessageCount; ++pI)
//    {
//        std::cout << pI << " : " << mPartialMessages[pI] << "\n";
//    }
    
    if( mComplete == true ) throw Exception("OSC Error: multi osc message has already set all its partial messages ", __FILE__, __FUNCTION__, __LINE__);
    if( pIndex > mTotalPartialMessageCount ) throw Exception("OSC Error: partial message index " + std::to_string(pIndex) + " exceeds expected total partial message count " + std::to_string(mTotalPartialMessageCount), __FILE__, __FUNCTION__, __LINE__);
    if( mPartialMessages[pIndex] )  throw Exception("OSC Error: partial message index " + std::to_string(pIndex) + " has already been assigned to a partial message ", __FILE__, __FUNCTION__, __LINE__);
    
    mPartialMessages[pIndex] = pPartialMessage;
    mCurrentPartialMessageCount++;
    
    if( mCurrentPartialMessageCount == mTotalPartialMessageCount ) mComplete = true;
    
}

/** \file dab_osc_message.h
 */

#ifndef _dab_osc_message_h_
#define _dab_osc_message_h_

#include <iostream>
#include <vector>
#include "dab_array.h"
#include "dab_osc_arg.h"


namespace dab
{
    
#pragma mark OscMessage definition
    
    class OscMessage
    {
    public:
        friend class OscUtils;
        
        OscMessage();
        OscMessage( const std::string& pAddress );
        OscMessage( const OscMessage& pOscMessage );
        ~OscMessage();
        
        const std::string& address() const;
        unsigned int argumentCount() const;
        std::string argumentTags() const;
        unsigned int arrayCount() const;
        std::string arrayTags() const;
        Array<unsigned int> arraySizes() const;
        
        const _OscArg* operator[]( unsigned int pArgIndex ) const;
        const std::vector< _OscArg* >& arguments() const;
        
        template<typename Type> bool argumentMatch( unsigned int pArgIndex ) const;
        bool argumentMatch( unsigned int pArgIndex, OscType pOscType ) const;
        bool argumentMatch( const std::initializer_list<OscType>& pOscTypes ) const;
        unsigned int valueCount(unsigned int pArgIndex) const;
        template<typename Type> void get( unsigned int pArgIndex, Type& pValue ) const;
        template<typename Type> void get( unsigned int pArgIndex, Type* pValues ) const;
        
        void clear();
        void setAddress( const std::string& pAddress );
        
        template<typename Type> void add( const Type& pValue );
        template<typename Type> void add( const Type& pValue, OscType pOscType );
        template<typename Type> void add( const Type* pValues, unsigned int pValueCount );
        template<typename Type> void add( const Type* pValues, unsigned int pValueCount, OscType pOscType );
        template<typename Type> void add( const std::initializer_list<Type>& pValues );
        template<typename Type> void add( const std::initializer_list<Type>& pValues, OscType pOscType  );
        
        unsigned int byteSize() const;
        operator std::string() const;
        
        friend std::ostream& operator<< (std::ostream & pOstream, const OscMessage& pMessage )
        {
            std::string dataString = pMessage;
            pOstream << dataString;
            return pOstream;
        }
        
    protected:
        std::string mAddress;
        std::vector< _OscArg* > mArguments;
    };
    
#pragma mark MultiOscMessage definition
    
    class MultiOscMessage
    {
    public:
        friend class OscUtils;
        
        MultiOscMessage();
        MultiOscMessage( const std::string& pAddress, unsigned long pId, unsigned int pPartialMessageCount );
        MultiOscMessage( const std::string& pAddress, unsigned long pId, unsigned int pPartialMessageCount, const std::string& pArrayTags, const Array<unsigned int>& pArraySizes );
        MultiOscMessage( const std::vector< std::shared_ptr< OscMessage> >& pPartialMessages, unsigned long pId, const std::string& pArrayTags, const Array<unsigned int>& pArraySizes );
        MultiOscMessage( std::shared_ptr<OscMessage> pOscMessage, unsigned long pId );
        ~MultiOscMessage();
        
        const std::string& address() const;
        unsigned long messageId() const;
        const std::string& arrayTags() const;
        const Array< unsigned int >& arraySizes() const;
        
        bool complete() const;
        std::shared_ptr<OscMessage> message() const;
        const std::vector< std::shared_ptr<OscMessage> >& partialMessages() const;
        
        void setArrayTags( const std::string& pArrayTags );
        void setArraySizes( const Array<unsigned int>& pArraySizes );
        void setPartialMessage( unsigned int pIndex, std::shared_ptr<OscMessage> pPartialMessage );
        
    protected:
        std::string mMessageAddress;
        unsigned long mMessageId;
        std::string mArrayTags;
        Array< unsigned int > mArraySizes;
        
        std::shared_ptr<OscMessage> mMessage;
        std::vector< std::shared_ptr<OscMessage> > mPartialMessages;
        
        unsigned int mTotalPartialMessageCount;
        unsigned int mCurrentPartialMessageCount;
        bool mComplete;
    };
    
#pragma mark OscMessage implementation
    
    template<typename Type>
    void
    OscMessage::add( const Type& pValue )
    {
        mArguments.push_back( new OscArg<Type>(pValue) );
    }
    
    template<typename Type>
    void
    OscMessage::add( const Type& pValue, OscType pOscType  )
    {
        mArguments.push_back( new OscArg<Type>(pValue, pOscType) );
    }
    
    template<typename Type>
    void
    OscMessage::add( const Type* pValues, unsigned int pValueCount )
    {
        mArguments.push_back( new OscArg<Type>(pValues, pValueCount) );
    }
    
    template<typename Type>
    void
    OscMessage::add( const Type* pValues, unsigned int pValueCount, OscType pOscType )
    {
        mArguments.push_back( new OscArg<Type>(pValues, pValueCount, pOscType) );
    }
    
    template<typename Type>
    void
    OscMessage::add( const std::initializer_list<Type>& pValues )
    {
        mArguments.push_back( new OscArg<Type>(pValues) );
    }
    
    template<typename Type>
    void
    OscMessage::add( const std::initializer_list<Type>& pValues, OscType pOscType  )
    {
        mArguments.push_back( new OscArg<Type>(pValues, pOscType) );
    }
    
    template<typename Type>
    bool
    OscMessage::argumentMatch( unsigned int pArgIndex ) const
    {
        if( pArgIndex >= mArguments.size() ) throw ("OSC Error: requested argument " + std::to_string(pArgIndex) + " exceeds number of arguments available " + std::to_string( mArguments.size() ), __FILE__, __FUNCTION__, __LINE__  );
        
        return OscUtils::get().oscType<Type>() == mArguments[pArgIndex]->oscType();
        
    }
    
    template<typename Type>
    void
    OscMessage::get( unsigned int pArgIndex, Type& pValue ) const
    {
        if( pArgIndex >= mArguments.size() ) throw ("OSC Error: requested argument " + std::to_string(pArgIndex) + " exceeds number of arguments available " + std::to_string( mArguments.size() ), __FILE__, __FUNCTION__, __LINE__  );
        
        try
        {
            _OscArg* arg = mArguments[pArgIndex];
            pValue = *(arg);
        }
        catch (Exception& e)
        {
            e += Exception("OSC Error: filed to retrieve osc argument " + std::to_string(pArgIndex), __FILE__, __FUNCTION__, __LINE__ );
            
            throw e;
        }
    }
    
    template<typename Type>
    void
    OscMessage::get( unsigned int pArgIndex, Type* pValues ) const
    {
        if( pArgIndex >= mArguments.size() ) throw ("OSC Error: requested argument " + std::to_string(pArgIndex) + " exceeds number of arguments available " + std::to_string( mArguments.size() ), __FILE__, __FUNCTION__, __LINE__  );
        
        try
        {
            _OscArg* arg = mArguments[pArgIndex];
            unsigned int valueCount = arg->valueCount();
            Type* values = *(arg);
            
            for(int i=0; i<valueCount; ++i) pValues[i] = values[i];
        }
        catch (Exception& e)
        {
            throw e += Exception("OSC Error: filed to retrieve osc argument " + std::to_string(pArgIndex), __FILE__, __FUNCTION__, __LINE__ );
        }
    }
    
};

#endif

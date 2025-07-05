/** \file dab_buffer.h
 *
 *  Created by Daniel Bisig on 26/5/2017.
 */

#pragma once

#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <typeindex>
#include "ofFileUtils.h"
#include "dab_exception.h"

namespace dab
{
    
#pragma mark _Buffer definition
    
class _Buffer
{
public:
    _Buffer();
    _Buffer( const _Buffer& pBuffer );
    _Buffer( unsigned int pValueCount, unsigned int pValueDim  );
    virtual ~_Buffer();
    
    virtual _Buffer* copy() const = 0;
    virtual _Buffer* create( unsigned int pValueCount, unsigned int pValueDim) const = 0;
    virtual _Buffer* create( const ofBuffer& pBuffer ) const = 0;
    
    virtual const _Buffer& operator=( const _Buffer& pBuffer ) = 0;
    virtual const _Buffer& operator=( const ofBuffer& pBuffer ) = 0;
    
    virtual std::type_index type() const = 0;
    bool sizeMatch( const _Buffer& pBuffer ) const;
    virtual bool typeMatch( const _Buffer& pBuffer ) const = 0;
    
    inline unsigned int valueCount() const;
    inline unsigned int totalValueCount() const;
    inline unsigned int valueDim() const;
    
    virtual void setSize( unsigned int pValueCount ) = 0;
    
    virtual operator ofBuffer() const = 0;
    virtual operator std::string() const = 0;
    
    friend std::ostream& operator<< (std::ostream & pOstream, const _Buffer& pBuffer)
    {
        pOstream << pBuffer.operator std::string();
        return pOstream;
    }
    
protected:
    unsigned int mValueCount;
    unsigned int mValueDim;
    unsigned int mTotalValueCount;
};
    
#pragma mark Buffer definition
        
template< typename Type >
class Buffer : public _Buffer
{
public:
    Buffer();
    Buffer( const Buffer<Type>& pBuffer );
    Buffer( const ofBuffer& pBuffer );
    template< typename Type2 > Buffer( const Buffer< Type2 >& pBuffer );
    Buffer( unsigned int pValueCount, unsigned int pValueDim );
    Buffer( unsigned int pValueCount, unsigned int pValueDim, Type pValue );
    Buffer( unsigned int pValueCount, unsigned int pValueDim, Type* pValues );
    template< typename Type2 > Buffer( unsigned int pValueCount, unsigned int pValueDim, Type2* pValues );
    ~Buffer();
    
    _Buffer* copy() const;
    _Buffer* create( unsigned int pValueCount, unsigned int pValueDim) const;
    _Buffer* create( const ofBuffer& pBuffer ) const;
    
    const _Buffer& operator=( const _Buffer& pBuffer );
    const _Buffer& operator=( const ofBuffer& pBuffer );
    const Buffer<Type>& operator=(const Buffer<Type>& pBuffer);
    template< typename Type2 > const Buffer<Type>& operator=( const Buffer<Type2>& pBuffer);

    std::type_index type() const;
    bool typeMatch( const _Buffer& pBuffer ) const;

    void setSize( unsigned int pValueCount );
    
    operator const Type*() const;
    operator Type*();
    operator ofBuffer() const;
    template<typename Type2> operator const Type2*() const;
    
    operator std::string() const;
    
    friend std::ostream& operator << ( std::ostream& pOstream, const Buffer<Type>& pBuffer)
    {
        pOstream << pBuffer.operator std::string();
        return pOstream;
    };
    
    Type* mValues; /**< \brief buffer values */
    
protected:
};

typedef Buffer<char> Bufferc;
typedef Buffer<unsigned char> Bufferuc;
typedef Buffer<int> Bufferi;
typedef Buffer<unsigned int> Bufferui;
typedef Buffer<float> Bufferf;
typedef Buffer<double> Bufferd;
    
#pragma mark _Buffer implementation
    
unsigned int
_Buffer::valueCount() const
{
    return mValueCount;
}
    
unsigned int
_Buffer::totalValueCount() const
{
    return mValueCount * mValueDim;
}
    
unsigned int
_Buffer::valueDim() const
{
    return mValueDim;
}

#pragma mark Buffer implementation

template< typename Type >
Buffer< Type >::Buffer()
: _Buffer(1,1)
{
    mValues = new Type[ mTotalValueCount ];
}

template< typename Type >
Buffer< Type >::Buffer( const Buffer<Type>& pBuffer )
: _Buffer(pBuffer)
{
    mValues = new Type[ mTotalValueCount ];
    for(int i=0; i<mTotalValueCount; ++i) mValues[i] = pBuffer.mValues[i];
}
    
template< typename Type >
Buffer<Type>::Buffer(const ofBuffer& pBuffer)
{
    mValueCount = pBuffer.size() / sizeof(Type);
    mValueDim = 1;
    mTotalValueCount = mValueCount;
    
    mValues = new Type[mValueCount];
    Type* data = (Type*)( pBuffer.getData() );
    for(unsigned int i=0; i<mValueCount; ++i) mValues[i] = data[i];
}

template< typename Type >
template< typename Type2 >
Buffer< Type >::Buffer( const Buffer<Type2>& pBuffer )
: _Buffer(pBuffer)
{
    mValues = new Type[ mTotalValueCount ];
    for(int i=0; i<mTotalValueCount; ++i) mValues[i] = static_cast<Type>( pBuffer.m[i] );
}

template< typename Type >
Buffer< Type >::Buffer( unsigned int pValueCount, unsigned int pValueDim )
: _Buffer(pValueCount,pValueDim)
{
    mValues = new Type[ mTotalValueCount ];
}

template< typename Type >
Buffer< Type >::Buffer( unsigned int pValueCount, unsigned int pValueDim, Type pValue )
: _Buffer(pValueCount,pValueDim)
{
    mValues = new Type[ mTotalValueCount ];
    for(int i=0; i<mTotalValueCount; ++i) mValues[i] = pValue;
}

template< typename Type >
Buffer< Type >::Buffer( unsigned int pValueCount, unsigned int pValueDim, Type* pValues )
: _Buffer(pValueCount,pValueDim)
{
    mValues = new Type[ mTotalValueCount ];
    for(int i=0; i<mTotalValueCount; ++i) mValues[i] = pValues[i];
}

template< typename Type >
template< typename Type2 >
Buffer< Type >::Buffer( unsigned int pValueCount, unsigned int pValueDim, Type2* pValues )
: _Buffer(pValueCount,pValueDim)
{
    mValues = new Type[ mTotalValueCount];
    for(int i=0; i<mTotalValueCount; ++i) mValues[i] = static_cast<Type>( pValues[i] );
}

template< typename Type >
Buffer< Type >::~Buffer()
{
    delete [] mValues;
}
    
template< typename Type >
_Buffer*
Buffer< Type >::copy() const
{
    return new Buffer<Type>(*this);
}
  
template< typename Type >
_Buffer*
Buffer< Type >::create( unsigned int pValueCount, unsigned int pValueDim) const
{
    return new Buffer<Type>(pValueCount, pValueDim);
}
    
template< typename Type >
_Buffer*
Buffer< Type >::create( const ofBuffer& pBuffer ) const
{
    return new Buffer<Type>(pBuffer);
}
    
    
template< typename Type >
const _Buffer&
Buffer< Type >::operator=(const _Buffer& pBuffer)
{
    std::string typeName = type().name();
    
    if( typeMatch(pBuffer) == false ) throw Exception( "Buffer Error: internal data type " + typeName + " doesn't match provided type " + pBuffer.type().name(), __FILE__, __FUNCTION__, __LINE__ );
    
    *this = (Buffer<Type>&)( pBuffer );
    
    return *this;
}
    
template<typename Type>
const _Buffer&
Buffer<Type>::operator=( const ofBuffer& pBuffer )
{
    *this = Buffer<Type>(pBuffer);
    
    return *this;
}
    
template< typename Type >
const Buffer<Type>&
Buffer< Type >::operator=(const Buffer<Type>& pBuffer)
{
    mValueCount = pBuffer.mValueCount;
    mValueDim = pBuffer.mValueDim;
    
    if( mTotalValueCount != pBuffer.mTotalValueCount ) 
    {
        delete [] mValues;
        mTotalValueCount = pBuffer.mTotalValueCount;
        mValues = new Type[ mTotalValueCount ];
    }
    
    for(int i=0; i<mTotalValueCount; ++i) mValues[i] = pBuffer.mValues[i];
    
    return *this;
}

template< typename Type >
template< typename Type2 >
const Buffer<Type>& 
Buffer< Type >::operator=( const Buffer<Type2>& pBuffer)
{
    mValueCount = pBuffer.mValueCount;
    mValueDim = pBuffer.mValueDim;
    
    if( mTotalValueCount != pBuffer.mTotalValueCount ) 
    {
        delete [] mValues;
        mTotalValueCount = pBuffer.mTotalValueCount;
        mValues = new Type[ mTotalValueCount ];
    }
    
    for(int i=0; i<mTotalValueCount; ++i) mValues[i] = static_cast<Type>( pBuffer.mValues[i] );
    
    return *this;
}
 
template< typename Type >
std::type_index
Buffer< Type >::type() const
{
    return std::type_index(typeid(Type));
}
    
template< typename Type >
bool
Buffer< Type >::typeMatch( const _Buffer& pBuffer ) const
{
    return type() == pBuffer.type();
}

template< typename Type >
void
Buffer< Type >::setSize( unsigned int pValueCount )
{
    if(mValueCount == pValueCount) return;
    
    int oldSize = mTotalValueCount;
    int newSize = pValueCount * mValueDim;
    
    Type* newValues = new Type[newSize];
    unsigned lowSize = oldSize < newSize ? oldSize : newSize;
    unsigned int i=0;
    
    for(; i<lowSize; ++i)
    {
        newValues[i] = mValues[i];
    }
    
    delete [] mValues;
    mValues = newValues;
    
    mValueCount = pValueCount;
    mTotalValueCount = newSize;
}
    
template<typename Type>
Buffer<Type>::operator const Type*() const
{
    return mValues;
}
  
template<typename Type>
Buffer<Type>::operator Type*()
{
    return mValues;
}
    
template<typename Type>
Buffer<Type>::operator ofBuffer() const
{
    return ofBuffer( (char*)(mValues), mTotalValueCount * sizeof(Type) );
}

template<typename Type>
template<typename Type2>
Buffer<Type>::operator const Type2*() const
{
    Type2* values = (Type2*)(mValues);
    return values;
}

template <typename Type>
Buffer<Type>::operator std::string() const
{
    std::stringstream ss;
    
    for(int v=0, i=0; v<mValueCount; ++v)
    {
        ss << "[ " << mValues[i++];
        
        for(int d=1; d<mValueDim; ++d, ++i) ss << ", " << mValues[i];
        
        ss << " ]";
    }
    
    return ss.str();
}	

};

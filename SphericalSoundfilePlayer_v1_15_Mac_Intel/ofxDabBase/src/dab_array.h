/** \file dab_array.h
 */

#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <typeindex>
#include <initializer_list>
#include "ofFileUtils.h"
#include "dab_exception.h"

namespace dab
{
    
#pragma mark _Array definition
    
    class _Array
    {
    public:
        _Array();
        _Array( unsigned int pSize );
        _Array( const _Array& pArray );
        virtual ~_Array();
        
        virtual std::type_index type() const = 0;
        bool sizeMatch( const _Array& pArray ) const;
        virtual bool typeMatch( const _Array& pArray ) const = 0;
        
        unsigned int size() const;
        virtual void setSize( unsigned int pSize ) = 0;
        
        virtual _Array* copy() const = 0;
        virtual _Array* create( unsigned int pSize ) const = 0;
        virtual _Array* create( const ofBuffer& pBuffer ) const = 0;
        
        virtual const _Array& operator=( const _Array& pArray ) = 0;
        virtual const _Array& operator=( const ofBuffer& pBuffer ) = 0;
        
        virtual operator ofBuffer() const = 0;
        virtual std::string info() const = 0;
        virtual operator std::string() const = 0;
        
        friend std::ostream& operator<< (std::ostream & pOstream, const _Array& pArray)
        {
            pOstream << std::string(pArray);
            return pOstream;
        }
        
    protected:
        unsigned int mSize;
    };
    
#pragma mark Array definition
    
    template < typename Type >
    class Array : public _Array
    {
    public:
        Array();
        Array( unsigned int pSize );
        Array( unsigned int pSize, const Type pValue );
        Array( unsigned int pSize, const Type* pValues );
        Array( const std::initializer_list<Type>& pValues );
        Array( const ofBuffer& pBuffer );
        Array( const Array<Type>& pArray );
        template<typename Type2> Array( const Array<Type2>& pArray );
        ~Array();
        
        std::type_index type() const;
        bool typeMatch( const _Array& pArray ) const;
        
        void setSize( unsigned int pSize );
        
        _Array* copy() const;
        _Array* create( unsigned int pSize ) const;
        _Array* create( const ofBuffer& pBuffer ) const;
        
        const _Array& operator=( const _Array& pArray );
        const _Array& operator=( const ofBuffer& pBuffer );
        
        const Array<Type>& operator=( const Array<Type>& pArray );
        const Array<Type>& operator=( const std::initializer_list<Type>& pValues );
        template<typename Type2> const Array<Type>& operator=( const Array<Type2>& pArray );
        
        Type& operator[]( unsigned int pIndex );
        const Type& operator[]( unsigned int pIndex ) const;
        
        bool operator==( const Array<Type>& pArray ) const;
        
        operator const Type*() const;
        operator Type*();
        operator ofBuffer() const;
        
        std::string info() const;
        operator std::string() const;
        template<typename Type2> operator const Type2*() const;
        
        friend std::ostream& operator<< (std::ostream & pOstream, const Array& pArray)
        {
            pOstream << pArray.operator std::string();
            return pOstream;
        }
        
    protected:
        Type* mValues;
    };
    
#pragma mark Array implementation
    
    template<typename Type>
    Array<Type>::Array()
    {
        mValues = new Type[1];
    }
    
    template<typename Type>
    Array<Type>::Array( unsigned int pSize )
    : _Array(pSize)
    {
        mValues = new Type[mSize];
    }
    
    template<typename Type>
    Array<Type>::Array( unsigned int pSize, const Type pValue )
    : _Array(pSize)
    {
        mValues = new Type[mSize];
        for(unsigned int i=0; i<pSize; ++i) mValues[i] = pValue;
    }
    
    template<typename Type>
    Array<Type>::Array( unsigned int pSize, const Type* pValues )
    : _Array(pSize)
    {
        mValues = new Type[mSize];
        for(unsigned int i=0; i<pSize; ++i) mValues[i] = pValues[i];
    }
    
    template<typename Type>
    Array<Type>::Array( const std::initializer_list<Type>& pValues )
    : _Array( pValues.size() )
    {
        mValues = new Type[mSize];
        int i=0;
        for(auto iter=pValues.begin(); iter!=pValues.end(); ++iter, ++i) mValues[i] = *iter;
    }
    
    template<typename Type>
    Array<Type>::Array( const ofBuffer& pBuffer )
    {
        mSize = pBuffer.size() / sizeof(Type);
        mValues = new Type[mSize];
        
        Type* values = (Type*)( pBuffer.getData() );
        for(unsigned int i=0; i<mSize; ++i) mValues[i] = values[i];
    }
    
    template<typename Type>
    Array<Type>::Array( const Array<Type>& pArray )
    : _Array(pArray)
    {
        mValues = new Type[mSize];
        for(unsigned int i=0; i<mSize; ++i) mValues[i] = pArray.mValues[i];
    }
    
    template<typename Type>
    template<typename Type2>
    Array<Type>::Array( const Array<Type2>& pArray )
    : _Array(pArray)
    {
        mValues = new Type[mSize];
        for(int i=0; i<_Array::mSize; ++i) mValues[i] = static_cast<Type>( pArray[i] );
    }
    
    template<typename Type>
    Array<Type>::~Array()
    {
        //std::cout << "delete " << mValues << "\n";
        
        delete [] mValues;
    }
    
    template<typename Type>
    std::type_index
    Array<Type>::type() const
    {
        return std::type_index(typeid(Type));
    }
    
    template<typename Type>
    bool
    Array<Type>::typeMatch( const _Array& pArray ) const
    {
        return type() == pArray.type();
    }
    
    template<typename Type>
    void
    Array<Type>::setSize( unsigned int pSize )
    {
        if( mSize == pSize ) return;
        
        unsigned int formerSize = pSize;
        Type* formerValues = mValues;
        
        mSize = pSize;
        mValues = new Type[pSize];
        
        unsigned int smallerSize = mSize > formerSize ? formerSize : mSize;
        
        for(int i=0; i<smallerSize; ++i) mValues[i] = formerValues[i];
        
        //std::cout << "delete " << formerValues << "\n";
        
        delete [] formerValues;
    }
    
    template<typename Type>
    _Array*
    Array<Type>::copy() const
    {
        return new Array(*this);
    }
    
    template<typename Type>
    _Array*
    Array<Type>::create( unsigned int pSize ) const
    {
        return new Array(pSize);
    }
    
    template<typename Type>
    _Array*
    Array<Type>::create( const ofBuffer& pBuffer ) const
    {
        return new Array(pBuffer);
    }
    
    template<typename Type>
    const _Array&
    Array<Type>::operator=( const _Array& pArray )
    {
        std::cout << "Array<Type>::operator=( const _Array& pArray ) throw (Exception)\n";
        
        std::string typeName = type().name();
        
        if( typeMatch(pArray) == false ) throw Exception( "Data Error: internal data type " + typeName + " doesn't match provided type " + pArray.type().name(), __FILE__, __FUNCTION__, __LINE__ );
            
            *this = (Array<Type>&)( pArray );
            
            return *this;
    }
    
    template<typename Type>
    const _Array&
    Array<Type>::operator=( const ofBuffer& pBuffer )
    {
        *this = Array<Type>(pBuffer);
        
        return *this;
    }
    
    template<typename Type>
    const Array<Type>&
    Array<Type>::operator=( const Array<Type>& pArray )
    {
        std::cout << "Array<Type>::operator=( const Array<Type>& pArray )\n";
        
        if( mSize != pArray.mSize ) setSize( pArray.mSize );
        for(int i=0; i<mSize; ++i) mValues[i] = pArray.mValues[i];
        
        return *this;
    }
    
    template<typename Type>
    const Array<Type>&
    Array<Type>::operator=( const std::initializer_list<Type>& pValues )
    {
        if( mSize != pValues.size() ) setSize( pValues.size() );
        
        int i=0;
        for(auto iter=pValues.begin(); iter!=pValues.end(); ++iter, ++i) mValues[i] = *iter;
        
        return *this;
    }
    
    template<typename Type>
    template<typename Type2>
    const Array<Type>&
    Array<Type>::operator=( const Array<Type2>& pArray )
    {
        unsigned int pSize = pArray.size();
        
        if( mSize != pSize ) setSize( pSize );
        
        const Type* pValues = pArray;
        
        for(int i=0; i<mSize; ++i) mValues[i] = static_cast<Type>(pArray[i]);
        
        return *this;
    }
    
    template<typename Type>
    Type&
    Array<Type>::operator[]( unsigned int pIndex )
    {
        if( pIndex >= mSize ) throw Exception( "Data Error: index " + std::to_string(pIndex) + " exceeds array size " + std::to_string(mSize), __FILE__, __FUNCTION__, __LINE__ );
            
            return mValues[pIndex];
    }
    
    template<typename Type>
    const Type&
    Array<Type>::operator[]( unsigned int pIndex ) const
    {
        if( pIndex >= mSize ) throw Exception( "Data Error: index " + std::to_string(pIndex) + " exceeds array size " + std::to_string(mSize), __FILE__, __FUNCTION__, __LINE__ );
            
            return mValues[pIndex];
    }
    
    template<typename Type>
    bool
    Array<Type>::operator==( const Array<Type>& pArray ) const
    {
        if( mSize != pArray.mSize ) return false;
        for(int d=0; d<mSize; ++d)
        {
            if(mValues[d] != pArray.mValues[d]) return false;
        }
        return true;
    }
    
    template<typename Type>
    Array<Type>::operator const Type*() const
    {
        return mValues;
    }
    
    template<typename Type>
    Array<Type>::operator Type*()
    {
        return mValues;
    }
    
    template<typename Type>
    Array<Type>::operator ofBuffer() const
    {
        return ofBuffer( (char*)(mValues), mSize * sizeof(Type) );
    }
    
    template<typename Type>
    template<typename Type2>
    Array<Type>::operator const Type2*() const
    {
        Type2* values = (Type2*)(mValues);
        
        return values;
    }

    template<typename Type>
    Array<Type>::operator std::string() const
    {
        return info();
    }
    
    template<typename Type>
    std::string
    Array<Type>::info() const
    {
        std::stringstream sstream;
        
        //        sstream << "type: " << type().name() << "\n";
        //        sstream << "size: " << mSize << "\n";
        //
        //        sstream << "values:";
        
        sstream << "[";
        
        for(unsigned int i=0; i<mSize; ++i)
        {
            sstream << " " << mValues[i];
        }
        sstream << " ]";
        
        return sstream.str();
    }
    
};

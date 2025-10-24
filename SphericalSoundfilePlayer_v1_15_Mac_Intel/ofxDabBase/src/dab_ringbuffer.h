/**	\file dab_ringbuffer.h
 *	\class dab::RingBuffer
 *	\brief template ringbuffer class
 *
 *   Ring Buffer Class.\n
 *   Elements must possess public default constructor, copy constructor and assigment operator.\n
 *
 *
 *  Created by John Flury on 22/03/07.
 *  Ported to OpenFrameworks by Daniel Bisig 05/10/2017
 *  Copyright 2007 John Flury All rights reserved.
 */

#pragma once

#include "dab_exception.h"

namespace dab
{

template<typename Type>
class RingBuffer
{
public:
    /**
     \brief parameter based constructor
     \param pSize number of ring buffer elements
     
     allocates ringer buffer and resets parameters
     elements are created via the default constructor
     */
    RingBuffer(unsigned int pSize);
    
    /**
     \brief parameter based constructor
     \param pSampleElement sample element
     \param pSize number of ring buffer elements
     
     allocates ringer buffer and resets parameters
     elements are created via copy constructor from the sample element
     */
    RingBuffer(const Type& pSampleElement, int pSize);
    
    /**
     \brief destructor
     */
    virtual ~RingBuffer();
    
    inline Type** elements();
    
    inline const Type** elements() const;
    
    /**
     \brief access nth element
     \param pIndex nth element's index
     
     0 = newest, nth = nth element (older)
     */
    inline Type& operator[] (unsigned int pIndex) throw (Exception);
    
    /**
     \brief access nth element
     \param pIndex nth element's index
     
     0 = newest, nth = nth element (older)
     */
    inline const Type& operator[] (unsigned int pIndex) const throw (Exception);
    
    /**
     \brief update ringbuffer with new element
     \param pElement new element
     */
    void update(const Type& pElement);
    
    /**
     \brief get the number of elements available in this ringbuffer
     */
    unsigned int size() const;
    
protected:
    /**
     \brief do not allow default constructor!
     */
    RingBuffer();
    
    /**
     \brief creates ring buffer elements
     */
    virtual void allocate();
    
    /**
     \brief creates ring buffer elements
     \param pSampleElement sample element
     */
    virtual void allocate(const Type& pSampleElement);
    
    /**
     \brief needs to be called, when contained elements
     */
    virtual void free();
    
    /**
     \brief array elements
     */
    Type** mElements;
    
    /**
     \brief number of elements
     */
    int mSize;
    
    /**
     \brief index to the newest element
     */
    int mIndex; //index to newest element
};

template <typename Type>
RingBuffer<Type>::RingBuffer()
: mSize(1)
, mIndex(0)
{
    allocate();
}

template <typename Type>
RingBuffer<Type>::RingBuffer(unsigned int pSize)
: mSize(pSize)
, mIndex(0)
{
    if(mSize == 0) mSize = 1;
    
    allocate();
}

template <typename Type>
RingBuffer<Type>::RingBuffer(const Type& pSampleElement, int pSize)
: mSize(pSize)
, mIndex(0)
{
    if(mSize == 0) mSize = 1;
    
    allocate(pSampleElement);
}

template <typename Type>
RingBuffer<Type>::~RingBuffer()
{
    free();
}
 
template <typename Type>
Type**
    RingBuffer<Type>::elements()
{
    return mElements;
}
   
template <typename Type>
const Type**
RingBuffer<Type>::elements() const
{
    return mElements;
}

template <typename Type>
void
RingBuffer<Type>::allocate()
{
    mElements = new Type*[mSize];
    
    for(unsigned int i=0; i<mSize; ++i) mElements[i] = new Type();
}

template <typename Type>
void
RingBuffer<Type>::allocate(const Type& pSampleElement)
{
    mElements = new Type*[mSize];
    
    for(unsigned int i=0; i<mSize; ++i) mElements[i] = new Type(pSampleElement);
}	

template <typename Type>
void
RingBuffer<Type>::free()
{
    for(unsigned int i=0; i<mSize; ++i) delete mElements[i];
    
    delete [] mElements;	
}

template <typename Type>
Type&
RingBuffer<Type>::operator[] (unsigned int pIndex) throw (Exception)
{
    if(pIndex >= mSize) throw Exception("DATA ERROR: index " + std::to_string(pIndex) + " exceeds size " + std::to_string(mSize) + " of ringbuffer", __FILE__, __FUNCTION__, __LINE__);
    
    int returnIndex = static_cast<int>(mIndex) - static_cast<int>(pIndex);
    return *(mElements[(returnIndex < 0 ? returnIndex+mSize : returnIndex)]);
}

template <typename Type>
const Type&
RingBuffer<Type>::operator[] (unsigned int pIndex) const throw (Exception)
{
    if(pIndex >= mSize) throw Exception("DATA ERROR: index " + std::to_string(pIndex) + " exceeds size " + std::to_string(mSize) + " of ringbuffer", __FILE__, __FUNCTION__, __LINE__);
        
    int returnIndex = static_cast<int>(mIndex) - static_cast<int>(pIndex);
    return *(mElements[(returnIndex < 0 ? returnIndex+mSize : returnIndex)]);
}

template <typename Type>
void 
RingBuffer<Type>::update(const Type& pNewElement) 
{
    unsigned int index = mIndex + 1;
    if (index>=mSize) index = 0; // wrap index to beginning
    
    (*mElements[index]) = pNewElement;
    
    mIndex = index;
}

template <typename Type>
unsigned int
RingBuffer<Type>::size() const
{
    return mSize;
}

};
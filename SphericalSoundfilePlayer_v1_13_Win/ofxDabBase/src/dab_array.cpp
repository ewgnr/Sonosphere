/** \file dab_array.cpp
*/

#include "dab_array.h"

using namespace dab;

#pragma mark _Array implementation

_Array::_Array()
    : mSize(0)
{}

_Array::_Array( unsigned int pSize )
    : mSize( pSize )
{}

_Array::_Array( const _Array& pArray )
    : mSize( pArray.mSize )
{}

_Array::~_Array()
{}

bool
_Array::sizeMatch( const _Array& pArray ) const
{
    return mSize == pArray.mSize;
}
    
unsigned int
_Array::size() const
{
    return mSize;
}
/** \file dab_buffer.cpp
*/

#include "dab_buffer.h"

using namespace dab;

_Buffer::_Buffer()
: mValueCount(0)
, mValueDim(0)
{}

_Buffer::_Buffer( const _Buffer& pBuffer)
: mValueCount(pBuffer.mValueCount)
, mValueDim(pBuffer.mValueDim)
{}

_Buffer::_Buffer( unsigned int pValueCount, unsigned int pValueDim )
: mValueCount(pValueCount)
, mValueDim(pValueDim)
{}

_Buffer::~_Buffer()
{}

bool
_Buffer::sizeMatch( const _Buffer& pBuffer ) const
{
    return mValueCount == pBuffer.mValueCount && mValueDim == pBuffer.mValueDim;
}


/** \file dab_value.cpp
*/

#include <sstream>
#include "dab_value.h"

using namespace dab;

AbstractValue::~AbstractValue()
{}

const std::type_index&
AbstractValue::type() const
{
    return mType;
}

bool
AbstractValue::typeMatch( const AbstractValue& pValue ) const
{
    return pValue.mType == mType;
}

std::string
AbstractValue::info() const
{
    std::stringstream ss;
    
    ss << "value type: " << mType.name();
    return ss.str();
}
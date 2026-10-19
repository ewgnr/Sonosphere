/** \file dab_osc_arg.cpp
*/

#include "dab_osc_arg.h"
#include "dab_osc_utils.h"
#include <stdio.h>
#include <string>

using namespace dab;

#pragma mark _OscArg implementation

_OscArg::_OscArg()
	: mOscType(OSC_TYPE_NONE)
{}

_OscArg::~_OscArg()
{}

OscType
_OscArg::oscType() const
{
	return mOscType;
}

#pragma mark OscArg implementation

template <>
OscArg<bool>::OscArg(const bool& pValue)
{
	mValueCount = 1;

	if (pValue == true) mOscType = OSC_TYPE_TRUE;
	else mOscType = OSC_TYPE_FALSE;

	mValues = new bool[1];
	mValues[0] = pValue;
}

template <>
unsigned int
OscArg<std::string>::valueCount() const
{
	//std::cout << "arg type " << OscUtils::get().typeName( mOscType ) << " valueCount " << mValueCount << "\n";

	return mValues[0].size();
}

template <>
unsigned int
OscArg<char>::typeSize() const
{
	return 4;
}

template <>
unsigned int
OscArg<std::string>::typeSize() const
{
	return 4;
}

template <>
unsigned int
OscArg<char>::byteCount() const
{
	//std::cout << "oscArg type " << OscUtils::get().typeName(mOscType) << " typeSize " << typeSize() << " valueCount " << mValueCount << "\n";

	return mValueCount * 4 + 8;
}

template <>
unsigned int
OscArg<std::string>::byteCount() const
{
	//std::cout << "oscArg type " << OscUtils::get().typeName(mOscType) << " typeSize " << typeSize() << " valueCount " << mValueCount << "\n";

	return mValues[0].size() * 4 + 8;

	//return (8 + (mValues[0].size() - 1 ) / 4 * 4);
}

template<>
_OscArg*
OscArg<std::string>::copy(unsigned int pStartIndex, unsigned int pValueCount) const
{
	if (mValues[0].size() < pStartIndex + pValueCount) throw Exception("OSC Error: copy range from " + std::to_string(pStartIndex) + " to " + std::to_string(pStartIndex + pValueCount) + " exceeds value count " + std::to_string(mValues[0].size()), __FILE__, __FUNCTION__, __LINE__);

	std::string subString = mValues[0].substr(pStartIndex, pValueCount);

	return new OscArg<std::string>(subString, mOscType);
}

template<>
void
OscArg<std::string>::append(const _OscArg& pArg)
{
	if (mOscType != pArg.oscType()) throw Exception("OSC Error: oscType mismatch,  required " + OscUtils::get().typeName(mOscType) + " received " + OscUtils::get().typeName(pArg.oscType()), __FILE__, __FUNCTION__, __LINE__);

	const OscArg<std::string>& typedArg = static_cast<const OscArg<std::string>&>(pArg);

	mValues[0] += typedArg.mValues[0];
}

template <>
void
OscArg<int64_t>::addToPacketStream(osc::OutboundPacketStream& pOstream) const
{
	pOstream << (osc::int64)mValues[0];
	//return pOstream;
}

template <>
void
OscArg<std::string>::addToPacketStream(osc::OutboundPacketStream& pOstream) const
{
	pOstream << mValues[0].c_str();
	//return pOstream;
}

template <>
std::string
OscArg<std::string>::info() const
{
	OscUtils& oscUtils = OscUtils::get();

	std::stringstream ss;

	ss << "Type " << oscUtils.typeName(mOscType) << "\n";
	ss << mValues[0];
	ss << "\n";

	return ss.str();
}

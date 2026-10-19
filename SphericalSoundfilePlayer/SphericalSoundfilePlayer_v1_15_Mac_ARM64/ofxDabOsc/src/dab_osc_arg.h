/** \file dab_osc_args.h
*/

#pragma once

#include <string>
#include <sstream>
#include <list>
#include "dab_osc_types.h"
#include "dab_osc_utils.h"

namespace dab
{

#pragma mark _OscArg definition

	class _OscArg
	{
	public:
		_OscArg();
		virtual ~_OscArg();

		OscType oscType() const;
		virtual unsigned int valueCount() const = 0;
		virtual unsigned int typeSize() const = 0;
		virtual unsigned int byteCount() const = 0;

		template<typename Type> operator Type&();
		template<typename Type> operator const Type&() const;
		template<typename Type> operator Type*();
		template<typename Type> operator const Type*() const;

		virtual _OscArg* copy() const = 0;
		virtual _OscArg* copy(unsigned int pStartIndex, unsigned int pValueCount) const = 0;
		virtual void append(const _OscArg& pTestClass) = 0;

		virtual void addToPacketStream(osc::OutboundPacketStream& pOstream) const = 0;

		//    friend osc::OutboundPacketStream& operator<< (osc::OutboundPacketStream& pOstream, const _OscArg& pOscArg )
		//    {
		//        pOstream = pOscArg.addToPacketStream(pOstream);
		//        
		//        return pOstream;
		//    }

		virtual std::string info() const = 0;

		friend std::ostream& operator<< (std::ostream & pOstream, const _OscArg& pOscArg)
		{
			std::string dataString = pOscArg.info();
			pOstream << dataString;
			return pOstream;
		}
	protected:
		OscType mOscType;
	};

#pragma mark OscArg definition

	template <typename Type>
	class OscArg : public _OscArg
	{
	public:
		OscArg();
		OscArg(const Type& pValue);
		OscArg(const Type& pValue, OscType pOscType);
		OscArg(const Type* pValues, unsigned int pValueCount);
		OscArg(const Type* pValues, unsigned int pValueCount, OscType pOscType);
		OscArg(const std::initializer_list<Type>& pValues);
		OscArg(const std::initializer_list<Type>& pValues, OscType pOscType);
		OscArg(const OscArg<Type>& pOscArg);
		~OscArg();

		unsigned int valueCount() const;
		unsigned int typeSize() const;
		unsigned int byteCount() const;

		operator Type&();
		operator const Type&() const;
		operator Type*();
		operator const Type*() const;

		const OscArg<Type>& operator=(const OscArg<Type>& pOscArg);

		_OscArg* copy() const;
		_OscArg* copy(unsigned int pStartIndex, unsigned int pValueCount) const;
		void append(const _OscArg& pOscArg);

		void addToPacketStream(osc::OutboundPacketStream& pOstream) const;

		std::string info() const;

		friend std::ostream& operator<< (std::ostream & pOstream, const OscArg& pOscArg)
		{
			std::string dataString = pOscArg.info();
			pOstream << dataString;
			return pOstream;
		}

	protected:
		Type* mValues;
		unsigned int mValueCount;
	};

	// declaration of specialisations
	template <> OscArg<bool>::OscArg(const bool& pValue);
	template <> unsigned int OscArg<std::string>::valueCount() const;
	template <> unsigned int OscArg<char>::typeSize() const;
	template <> unsigned int OscArg<std::string>::typeSize() const;
	template <> unsigned int OscArg<char>::byteCount() const;
	template <> unsigned int OscArg<std::string>::byteCount() const;
	template<> _OscArg* OscArg<std::string>::copy(unsigned int pStartIndex, unsigned int pValueCount) const;
	template<> void OscArg<std::string>::append(const _OscArg& pArg);
	template <> void OscArg<int64_t>::addToPacketStream(osc::OutboundPacketStream& pOstream) const;
	template <> void OscArg<std::string>::addToPacketStream(osc::OutboundPacketStream& pOstream) const;
	template <> std::string OscArg<std::string>::info() const;

#pragma mark _OscArg implementation

	template<typename Type>
	_OscArg::operator Type&()
	{
		OscUtils& oscUtils = OscUtils::get();
		OscType pOscTape = oscUtils.oscType<Type>();

		//std::cout << "_OscArg::operator Type&() " << oscUtils.typeName(mOscType) << " " << oscUtils.typeName( pOscTape ) << "\n";

		if (mOscType != pOscTape) throw Exception("OSC Error: osc type mismatch: available " + oscUtils.typeName(mOscType) + " requested " + oscUtils.typeName(pOscTape), __FILE__, __FUNCTION__, __LINE__);

		return static_cast< OscArg<Type>* >(this)->operator Type&();
	}

	template<typename Type>
	_OscArg::operator const Type&() const
	{
		OscUtils& oscUtils = OscUtils::get();
		OscType pOscTape = oscUtils.oscType<Type>();

		//std::cout << "_OscArg::operator Type&() " << oscUtils.typeName(mOscType) << " " << oscUtils.typeName( pOscTape ) << "\n";

		if (mOscType != pOscTape) throw Exception("OSC Error: osc type mismatch: available " + oscUtils.typeName(mOscType) + " requested " + oscUtils.typeName(pOscTape), __FILE__, __FUNCTION__, __LINE__);

		return static_cast< const OscArg<Type>* >(this)->operator const Type&();
	}

	template<typename Type>
	_OscArg::operator Type*()
	{
		OscUtils& oscUtils = OscUtils::get();
		OscType pOscTape = oscUtils.oscType<Type*>();

		if (mOscType != pOscTape) throw Exception("OSC Error: osc type mismatch: available " + oscUtils.typeName(mOscType) + " requested " + oscUtils.typeName(pOscTape), __FILE__, __FUNCTION__, __LINE__);

		return static_cast< OscArg<Type>* >(this)->operator Type*();
	}

	template<typename Type>
	_OscArg::operator const Type*() const
	{
		OscUtils& oscUtils = OscUtils::get();
		OscType pOscTape = oscUtils.oscType<Type*>();

		if (mOscType != pOscTape) throw Exception("OSC Error: osc type mismatch: available " + oscUtils.typeName(mOscType) + " requested " + oscUtils.typeName(pOscTape), __FILE__, __FUNCTION__, __LINE__);

		return static_cast< const OscArg<Type>* >(this)->operator const Type*();
	}

#pragma mark OscArg implementation

	template <typename Type>
	OscArg<Type>::OscArg()
	{
		mValueCount = 1;
		mOscType = OscUtils::get().oscType<Type>();

		mValues = new Type[1];
	}

	template <typename Type>
	OscArg<Type>::OscArg(const Type& pValue)
	{
		mValueCount = 1;
		mOscType = OscUtils::get().oscType<Type>();

		mValues = new Type[1];
		mValues[0] = pValue;
	}

	template <typename Type>
	OscArg<Type>::OscArg(const Type& pValue, OscType pOscType)
	{
		mValueCount = 1;
		mOscType = pOscType;

		mValues = new Type[1];
		mValues[0] = pValue;
	}

	template<typename Type>
	OscArg<Type>::OscArg(const Type* pValues, unsigned int pValueCount)
	{
		mValueCount = pValueCount;
		mOscType = OscUtils::get().oscType<Type*>();

		mValues = new Type[mValueCount];
		for (int i = 0; i<mValueCount; ++i) mValues[i] = pValues[i];
	}

	template<typename Type>
	OscArg<Type>::OscArg(const Type* pValues, unsigned int pValueCount, OscType pOscType)
	{
		mValueCount = pValueCount;
		mOscType = pOscType;

		mValues = new Type[mValueCount];
		for (int i = 0; i<mValueCount; ++i) mValues[i] = pValues[i];
	}

	template<typename Type>
	OscArg<Type>::OscArg(const std::initializer_list<Type>& pValues)
	{
		mValueCount = pValues.size();
		mOscType = OscUtils::get().oscType<Type*>();

		mValues = new Type[mValueCount];
		int i = 0;
		for (auto iter = pValues.begin(); iter != pValues.end(); ++iter, ++i) mValues[i] = *iter;
	}

	template<typename Type>
	OscArg<Type>::OscArg(const std::initializer_list<Type>& pValues, OscType pOscType)
	{
		mValueCount = pValues.size();
		mOscType = pOscType;

		mValues = new Type[mValueCount];
		int i = 0;
		for (auto iter = pValues.begin(); iter != pValues.end(); ++iter, ++i) mValues[i] = *iter;
	}

	template<typename Type>
	OscArg<Type>::OscArg(const OscArg<Type>& pOscArg)
	{
		mValueCount = pOscArg.mValueCount;
		mOscType = pOscArg.mOscType;

		mValues = new Type[mValueCount];
		for (int i = 0; i<mValueCount; ++i) mValues[i] = pOscArg.mValues[i];
	}

	template <typename Type>
	OscArg<Type>::~OscArg()
	{
		delete[] mValues;
	}

	template <typename Type>
	unsigned int
		OscArg<Type>::valueCount() const
	{
		return mValueCount;
	}

	template <typename Type>
	unsigned int
		OscArg<Type>::typeSize() const
	{
		return sizeof(Type);
	}

	template <typename Type>
	unsigned int
		OscArg<Type>::byteCount() const
	{
		//std::cout << "oscArg type " << OscUtils::get().typeName(mOscType) << " typeSize " << typeSize() << " valueCount " << mValueCount << "\n";

		return typeSize() * (mValueCount + 1);
	}

	template<typename Type>
	OscArg<Type>::operator Type&()
	{
		return mValues[0];
	}

	template<typename Type>
	OscArg<Type>::operator const Type&() const
	{
		return mValues[0];
	}

	template<typename Type>
	OscArg<Type>::operator Type*()
	{
		return mValues;
	}

	template<typename Type>
	OscArg<Type>::operator const Type*() const
	{
		return mValues;
	}

	template <typename Type>
	const OscArg<Type>&
		OscArg<Type>::operator=(const OscArg<Type>& pOscArg)
	{
		delete[] mValues;
		mValueCount = pOscArg.mValueCount;
		mValues = new Type[mValueCount];

		for (int i = 0; i<mValueCount; ++i) mValues[i] = pOscArg.mValues[i];

		mOscType = pOscArg.mOscType;

		return *this;
	}

	template<typename Type>
	_OscArg*
		OscArg<Type>::copy() const
	{
		return new OscArg(*this);
	}

	template<typename Type>
	_OscArg*
		OscArg<Type>::copy(unsigned int pStartIndex, unsigned int pValueCount) const
	{
		if (mValueCount < pStartIndex + pValueCount) throw Exception("OSC Error: copy range from " + std::to_string(pStartIndex) + " to " + std::to_string(pStartIndex + pValueCount) + " exceeds value count " + std::to_string(mValueCount), __FILE__, __FUNCTION__, __LINE__);

		Type* copyValues = new Type[pValueCount];

		for (int i = 0, j = pStartIndex; i<pValueCount; ++i, ++j)
		{
			copyValues[i] = mValues[j];
		}

		OscArg<Type>* newArg = new OscArg<Type>(copyValues, pValueCount, mOscType);

		delete[] copyValues;

		return newArg;
	}

	template<typename Type>
	void
		OscArg<Type>::append(const _OscArg& pArg)
	{
		if (mOscType != pArg.oscType()) throw Exception("OSC Error: oscType mismatch,  required " + OscUtils::get().typeName(mOscType) + " received " + OscUtils::get().typeName(pArg.oscType()), __FILE__, __FUNCTION__, __LINE__);

		const OscArg<Type>& typedArg = static_cast<const OscArg<Type>& >(pArg);

		Type* oldValues = mValues;
		unsigned int oldValueCount = mValueCount;

		mValueCount += typedArg.mValueCount;
		mValues = new Type[mValueCount];

		for (int i = 0; i<oldValueCount; ++i) mValues[i] = oldValues[i];
		for (int i = oldValueCount, j = 0; i < mValueCount; ++i, ++j) mValues[i] = typedArg.mValues[j];

		delete[] oldValues;
	}

	template <typename Type>
	void
		OscArg<Type>::addToPacketStream(osc::OutboundPacketStream& pOstream) const
	{
		if (mValueCount == 1)
		{
			pOstream << mValues[0];
		}
		else
		{
			osc::Blob blob((void*)mValues, sizeof(Type) * mValueCount);
			pOstream << blob;
		}
	}

	template <>
	void
		OscArg<int64_t>::addToPacketStream(osc::OutboundPacketStream& pOstream) const;

	template <>
	void
		OscArg<std::string>::addToPacketStream(osc::OutboundPacketStream& pOstream) const;

	template <typename Type>
	std::string
		OscArg<Type>::info() const
	{
		OscUtils& oscUtils = OscUtils::get();

		std::stringstream ss;

		ss << "Type " << oscUtils.typeName(mOscType) << "\n";

		for (int i = 0; i<mValueCount; ++i) ss << mValues[i] << " ";
		ss << "\n";

		return ss.str();
	}

};

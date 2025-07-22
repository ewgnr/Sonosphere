/** \file dab_index_hash.h
*/

#pragma once

#include <iostream>
#include <vector>
#include <map>
#include "dab_exception.h"
#include "dab_value.h"

namespace dab
{

template<typename KeyType, typename ValueType>
class IndexMap
{
public:
    IndexMap();
    ~IndexMap();
    
    /**
     \brief return vector of data
     \return vector of data
     */
    std::vector<ValueType>& vector();
    
    /**
     \brief return values
     \return values
     */
    std::vector<ValueType>& values();
    
    /**
     \brief return values
     \return values
     */
    const std::vector<ValueType>& values() const;
    
    /**
     \brief return keys
     \return keys
     */
    const std::vector<KeyType>& keys() const;
    
    /**
     \brief return map table of data
     \return map table of data
     */
    std::map<KeyType, ValueType>& map();
    
    /**
     \brief return map table of data
     \return map table of data
     */
    const std::map<KeyType, ValueType>& map() const;
    
    /**
     \brief returns number of elements stored
     \return number of elements stored
     */
	unsigned int size() const;
	
	/**
     \brief check whether hash table contains key
     \param pKey key to look for
     \return true if key is found, false otherwise
     */
	bool contains(KeyType pKey) const;
	
	/**
     \brief return index for key
     \param pKey key to look for
     \return index
     \exception Exception key not found
     */
	unsigned int index(KeyType pKey) const;
    
	/**
     \brief return key
     \param pIndex index of element
     \return key
     \exception Exception index out of range
     */
	inline const KeyType& key(unsigned int pIndex) const;

	/**
     \brief return value
     \param pIndex index of element
     \return value
     \exception Exception index out of range
     */
	inline const ValueType& value(unsigned int pIndex) const;
    
	/**
     \brief return value
     \param pIndex index of element
     \return value
     \exception Exception index out of range
     */
	inline ValueType& value(unsigned int pIndex);

    /**
     \brief access element
     \param pIndex index of element
     \return element
     \exception Exception index out of range
     */
	inline const ValueType& operator[] (unsigned int pIndex) const;
    
	/**
     \brief access element
     \param pIndex index of element
     \return element
     \exception Exception index out of range
     */
	inline ValueType& operator[] (unsigned int pIndex);

	/**
     \brief access element
     \param pKey key of element
     \return element
     \exception Exception key not found
     */
	inline const ValueType& operator[] (const KeyType& pKey) const;
    
	/**
     \brief access element
     \param pKey key of element
     \return element
     \exception Exception key not found
     */
	inline ValueType& operator[] (const KeyType& pKey);

	/**
     \brief add element
     \param pKey storage key
     \param pValue element value
     \exception Exception key alreay stored
     */
	void add(const KeyType& pKey, const ValueType& pValue);

	/**
     \brief insert element
     \param pKey storage key
     \param pValue element value
     \exception Exception key alreay stored
     
     same as add
     */
	void insert(const KeyType& pKey, const ValueType& pValue);

	/**
	 \brief insert element
	 \param pKey storage key
	 \param pValue element value
	 \param pIndex insert position index
	 \exception Exception key alreay stored
	 */
	void insert(const KeyType& pKey, const ValueType& pValue, unsigned int pIndex );

	/**
	 \brief move element
	 \param pFromIndex from index position
	 \param pToIndex to indey position
	 \exception Exception index out of bounds
	 */
	void move( unsigned int pFromIndex, unsigned int pToIndex );

	/**
     \brief replace element
     \param pKey storage key
     \param pValue element value
     \exception Exception key not found
     */
	void replace(const KeyType& pKey, const ValueType& pValue);
    
	/**
     \brief replace element
     \param pIndex index of element
     \param pValue element value
     \exception Exception index out of range
     */
	void replace(unsigned int pIndex, const ValueType& pValue);
    
	/**
     \brief set element
     \param pKey storage key
     \param pValue element value
     
     if key is found: element is replaced
     if key is not found: element is added
     */
	void set(const KeyType& pKey, const ValueType& pValue);
    
	/**
     \brief set element
     \param pIndex index of element
     \param pValue element value
     \exception Exception index out of range
     */
	void set(unsigned int pIndex, const ValueType& pValue);
    
	/**
     \brief remove element
     \param pKey storage key
     \exception Exception key not found
     */
	void remove(const KeyType& pKey);
    
	/**
     \brief remove element
     \param pIndex index of element
     \exception Exception index out of range
     */
	void remove(unsigned int pIndex);

	/**
     \brief remove all elements
     */
	void clear();
    
    /**
	 \brief return polyLine as string
	 \returns polyLine as string
	 */
	operator std::string() const;
	
	/**
	 \brief print polyline
	 \param pOstream output stream
	 \param pPolyLine polyLine
	 \returns output stream
	 */
	friend std::ostream& operator << ( std::ostream& pOstream, const IndexMap& pIndexMap)
	{
		pOstream << std::string( pIndexMap );
		return pOstream;
	};

    
protected:
    /**
     \brief vector of data
     */
    std::vector<ValueType> mValueVector;
    
	/**
     \brief vector of keys
     */
    std::vector<KeyType> mKeyVector;
	
	/**
     \brief hash table of data
     */
    std::map<KeyType, ValueType> mMap;
};
    
template <typename KeyType, typename ValueType>
IndexMap<KeyType, ValueType>::IndexMap()
{}
    
template <typename KeyType, typename ValueType>
IndexMap<KeyType, ValueType>::~IndexMap()
{
    mValueVector.clear();
    mKeyVector.clear();
    mMap.clear();
}
    
template <typename KeyType, typename ValueType>
std::vector<ValueType>&
IndexMap<KeyType, ValueType>::vector()
{
    return mValueVector;
}
    
template <typename KeyType, typename ValueType>
std::vector<ValueType>&
IndexMap<KeyType, ValueType>::values()
{
    return mValueVector;
}
    
template <typename KeyType, typename ValueType>
const std::vector<ValueType>&
IndexMap<KeyType, ValueType>::values() const
{
    return mValueVector;
}
    
template <typename KeyType, typename ValueType>
const std::vector<KeyType>&
IndexMap<KeyType, ValueType>::keys() const
{
    return mKeyVector;
}
    
template <typename KeyType, typename ValueType>
std::map<KeyType, ValueType>&
IndexMap<KeyType, ValueType>::map()
{
    return mMap;
}
    
template <typename KeyType, typename ValueType>
const std::map<KeyType, ValueType>&
IndexMap<KeyType, ValueType>::map() const
{
    return mMap;
}
    
template <typename KeyType, typename ValueType>
unsigned int
IndexMap<KeyType, ValueType>::size() const
{
    return mValueVector.size();
}

template <typename KeyType, typename ValueType>
bool
IndexMap<KeyType, ValueType>::contains(KeyType pKey) const
{
    return mMap.find(pKey) != mMap.end();
}
    
template <typename KeyType, typename ValueType>
unsigned int
IndexMap<KeyType, ValueType>::index(KeyType pKey) const
{
    unsigned int index = 0;
    auto keyIter = std::find(mKeyVector.begin(), mKeyVector.end(), pKey);
    
    if(keyIter != mKeyVector.end()) return keyIter - mKeyVector.begin();
    else throw(Exception( "DATA ERROR: hash key " + Value<KeyType>(pKey).info() + " not found", __FILE__, __FUNCTION__, __LINE__ ));
}
    
template <typename KeyType, typename ValueType>
const KeyType&
IndexMap<KeyType, ValueType>::key(unsigned int pIndex) const
{
    if(pIndex >= mKeyVector.size()) throw(Exception( "DATA ERROR: index " + std::to_string(pIndex) + " exceeds size of IndexMap " + std::to_string(mKeyVector.size()), __FILE__, __FUNCTION__, __LINE__ ));
    return mKeyVector[pIndex];
}
    
template <typename KeyType, typename ValueType>
const ValueType&
IndexMap<KeyType, ValueType>::value(unsigned int pIndex) const
{
    if(pIndex >= mValueVector.size()) throw(Exception( "DATA ERROR: index " + std::to_string(pIndex) + " exceeds size of IndexMap " + std::to_string(mValueVector.size()), __FILE__, __FUNCTION__, __LINE__ ));
    return mValueVector[pIndex];
}
    
template <typename KeyType, typename ValueType>
ValueType&
IndexMap<KeyType, ValueType>::value(unsigned int pIndex)
{
    if(pIndex >= mValueVector.size()) throw(Exception( "DATA ERROR: index " + std::to_string(pIndex) + " exceeds size of IndexMap " + std::to_string(mValueVector.size()), __FILE__, __FUNCTION__, __LINE__ ));
    return mValueVector[pIndex];
}
    
template <typename KeyType, typename ValueType>
const ValueType&
IndexMap<KeyType, ValueType>::operator[] (unsigned int pIndex) const
{
    if(pIndex >= mValueVector.size()) throw(Exception( "DATA ERROR: index " + std::to_string(pIndex) + " exceeds size of IndexMap " + std::to_string(mValueVector.size()), __FILE__, __FUNCTION__, __LINE__ ));
    return mValueVector[pIndex];
}

template <typename KeyType, typename ValueType>
ValueType&
IndexMap<KeyType, ValueType>::operator[] (unsigned int pIndex)
{
    if(pIndex >= mValueVector.size()) throw(Exception( "DATA ERROR: index " + std::to_string(pIndex) + " exceeds size of IndexMap " + std::to_string(mValueVector.size()), __FILE__, __FUNCTION__, __LINE__ ));
    return mValueVector[pIndex];
}
    
template <typename KeyType, typename ValueType>
const ValueType&
IndexMap<KeyType, ValueType>::operator[] (const KeyType& pKey) const
{
    if(mMap.find(pKey) == mMap.end()) throw(Exception( "DATA ERROR: map key " + Value<KeyType>(pKey).info() + "not found", __FILE__, __FUNCTION__, __LINE__ ));
    return mMap.at(pKey);
}
    
template <typename KeyType, typename ValueType>
ValueType&
IndexMap<KeyType, ValueType>::operator[] (const KeyType& pKey)
{
    if(mMap.find(pKey) == mMap.end()) throw(Exception( "DATA ERROR: map key "  + Value<KeyType>(pKey).info() +  " not found", __FILE__, __FUNCTION__, __LINE__ ));
    return mMap[pKey];
}
    
template <typename KeyType, typename ValueType>
void
IndexMap<KeyType, ValueType>::add(const KeyType& pKey, const ValueType& pValue)
{
    if(mMap.find(pKey) != mMap.end()) throw(Exception( "DATA ERROR: map key "  + Value<KeyType>(pKey).info() +  " already stored", __FILE__, __FUNCTION__, __LINE__ ));
    
    mValueVector.push_back(pValue);
    mKeyVector.push_back(pKey);
    mMap[pKey] = pValue;
}
    
template <typename KeyType, typename ValueType>
void
IndexMap<KeyType, ValueType>::insert(const KeyType& pKey, const ValueType& pValue)
{
    if(mMap.find(pKey) != mMap.end()) throw(Exception( "DATA ERROR: map key "  + Value<KeyType>(pKey).info() +  " already stored", __FILE__, __FUNCTION__, __LINE__ ));

    mValueVector.push_back(pValue);
    mKeyVector.push_back(pKey);
    mMap[pKey] = pValue;
}
    
template <typename KeyType, typename ValueType>
void
IndexMap<KeyType, ValueType>::insert(const KeyType& pKey, const ValueType& pValue, unsigned int pIndex )
{
    if(mMap.find(pKey) != mMap.end()) throw(Exception( "DATA ERROR: map key "  + Value<KeyType>(pKey).info() +  " already stored", __FILE__, __FUNCTION__, __LINE__ ));
    if(pIndex >= mValueVector.size()) throw(Exception( "DATA ERROR: index " + std::to_string(pIndex) + " exceeds size of IndexMap " + std::to_string(mValueVector.size()), __FILE__, __FUNCTION__, __LINE__ ));
    
    
    mValueVector.insert(mValueVector.begin() + pIndex, pValue);
    mKeyVector.insert(mKeyVector.begin() + pIndex, pKey);
    mMap[pKey] = pValue;
}
    
template <typename KeyType, typename ValueType>
void
IndexMap<KeyType, ValueType>::move(unsigned int pFromIndex, unsigned int pToIndex)
{
    // TODO: check that it is really correct what this function is doing
    
    if(pFromIndex >= mValueVector.size()) throw(Exception( "DATA ERROR: fromIndex " + std::to_string(pFromIndex) + " exceeds size of IndexMap " + std::to_string(mValueVector.size()), __FILE__, __FUNCTION__, __LINE__ ));
    if(pToIndex >= mValueVector.size()) throw(Exception( "DATA ERROR: toIndex " + std::to_string(pToIndex) + " exceeds size of IndexMap " + std::to_string(mValueVector.size()), __FILE__, __FUNCTION__, __LINE__ ));
    
    ValueType& _value = mValueVector[pFromIndex];
    KeyType& _key = mKeyVector[pFromIndex];
    
    mValueVector.erase(mValueVector.begin() + pFromIndex);
    mKeyVector.erase(mKeyVector.begin() + pFromIndex);
    
    mValueVector.insert(mValueVector.begin() + pToIndex, _value);
    mKeyVector.insert(mKeyVector.begin() + pToIndex, _key);
}
    
template <typename KeyType, typename ValueType>
void
IndexMap<KeyType, ValueType>::replace(const KeyType& pKey, const ValueType& pValue)
{
    if(mMap.find(pKey) == mMap.end()) throw(Exception( "DATA ERROR: map key "  + Value<KeyType>(pKey).info() +  " not found", __FILE__, __FUNCTION__, __LINE__ ));
    
    ValueType& origValue = mMap[pKey];
    mMap[pKey] = pValue;
    
    try
    {
        int keyIndex = index(pKey);
        mValueVector[keyIndex] = pValue;
    }
    catch(Exception& e)
    {
        e += Exception("DATA ERROR: failed to replace value for key " + Value<KeyType>(pKey).info(), __FILE__, __FUNCTION__, __LINE__);
        throw e;
    }
}
    
template <typename KeyType, typename ValueType>
void
IndexMap<KeyType, ValueType>::replace(unsigned int pIndex, const ValueType& pValue)
{
    if(pIndex >= mValueVector.size()) throw(Exception( "DATA ERROR: index " + std::to_string(pIndex) + " exceeds size of IndexMap " + std::to_string(mValueVector.size()), __FILE__, __FUNCTION__, __LINE__ ));
    
    ValueType& origValue = mValueVector[pIndex];
    mValueVector[pIndex] = pValue;
    
    KeyType& key = mKeyVector[pIndex];
    auto mapIter = mMap.find(key);
    mapIter->second = pValue;
}
    
template <typename KeyType, typename ValueType>
void
IndexMap<KeyType, ValueType>::set(const KeyType& pKey, const ValueType& pValue)
{
    try
    {
        if(mMap.find(pKey) != mMap.end()) replace(pKey, pValue);
        else add(pKey, pValue);
    }
    catch(Exception& e)
    {
        e += Exception("DATA ERROR: failed to set key " + Value<KeyType>(pKey).info() + " to value " + Value<ValueType>(pValue).info(), __FILE__, __FUNCTION__, __LINE__);
        throw e;
    }
}
    
template <typename KeyType, typename ValueType>
void
IndexMap<KeyType, ValueType>::set(unsigned int pIndex, const ValueType& pValue)
{
    if(pIndex >= mValueVector.size()) throw(Exception( "DATA ERROR: index " + std::to_string(pIndex) + " exceeds size of IndexMap " + std::to_string(mValueVector.size()), __FILE__, __FUNCTION__, __LINE__ ));

    try
    {
        replace(pIndex, pValue);
    }
    catch(Exception& e)
    {
        e += Exception("DATA ERROR: failed to set index " + std::to_string(pIndex)+ " to value " + Value<ValueType>(pValue).info(), __FILE__, __FUNCTION__, __LINE__);
        throw e;
    }
}
    
template <typename KeyType, typename ValueType>
void
IndexMap<KeyType, ValueType>::remove(const KeyType& pKey)
{
    if(mMap.find(pKey) == mMap.end()) throw(Exception( "DATA ERROR: map key "  + Value<KeyType>(pKey).info() +  " not found", __FILE__, __FUNCTION__, __LINE__ ));
    
    mMap.erase(pKey);
    
    try
    {
        int keyIndex = index(pKey);
        mValueVector.erase( mValueVector.begin() + keyIndex );
        mKeyVector.erase( mKeyVector.begin() + keyIndex );
    }
    catch(Exception& e)
    {
        e += Exception("DATA ERROR: failed to remove key " + Value<KeyType>(pKey).info(), __FILE__, __FUNCTION__, __LINE__);
        throw e;
    }
}
    
template <typename KeyType, typename ValueType>
void
IndexMap<KeyType, ValueType>::remove(unsigned int pIndex)
{
    if(pIndex >= mValueVector.size()) throw(Exception( "DATA ERROR: index " + std::to_string(pIndex) + " exceeds size of IndexMap " + std::to_string(mValueVector.size()), __FILE__, __FUNCTION__, __LINE__ ));
 
    try
    {
       const KeyType& _key = key(pIndex);

        mMap.erase(_key);
        mValueVector.erase( mValueVector.begin() + pIndex );
        mKeyVector.erase( mKeyVector.begin() + pIndex );
    }
    catch(Exception& e)
    {
        e += Exception("DATA ERROR: failed to remove index " + std::to_string(pIndex), __FILE__, __FUNCTION__, __LINE__);
        throw e;
    }
}
    
template <typename KeyType, typename ValueType>
void
IndexMap<KeyType, ValueType>::clear()
{
    mValueVector.clear();
    mKeyVector.clear();
    mMap.clear();
}
    
template <typename KeyType, typename ValueType>
IndexMap<KeyType, ValueType>::operator std::string() const
{
    std::stringstream stream;
    
    std::cout << "IndexMap:\n";
    int valueCount = mValueVector.size();
    for(int vI=0; vI<valueCount; ++vI)
    {
        const KeyType& _key = key(vI);
        const ValueType& _value = mMap.at(_key);
        
        stream << "index " << vI << " key (" << Value<KeyType>(_key).info() << ") value (" << Value<ValueType>(_value).info() << ")\n";
    }
    
    return stream.str();
}

    
};

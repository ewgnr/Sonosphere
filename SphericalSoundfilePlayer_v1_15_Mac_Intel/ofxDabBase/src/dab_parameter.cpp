/** \file dab_parameter.cpp
*/

#include "dab_parameter.h"

using namespace dab;

#pragma mark AbstractParameter implementation

AbstractParameter::AbstractParameter(unsigned int pDataDim)
: mValue(nullptr)
, mMinValue(nullptr)
, mMaxValue(nullptr)
, mDataDim(pDataDim)
{}

AbstractParameter::AbstractParameter(const std::string& pName, unsigned int pDataDim)
: mName(pName)
, mValue(nullptr)
, mMinValue(nullptr)
, mMaxValue(nullptr)
, mDataDim(pDataDim)
{}

AbstractParameter::AbstractParameter(const AbstractParameter& pControl)
: mName(pControl.mName)
, mValue(nullptr)
, mMinValue(nullptr)
, mMaxValue(nullptr)
, mDataDim(pControl.mDataDim)
{}

AbstractParameter::~AbstractParameter()
{}

const std::string
AbstractParameter::name() const
{
    return mName;
}

unsigned int
AbstractParameter::dataDim() const
{
    return mDataDim;
}

bool
AbstractParameter::hasValue() const
{
    return mValue != nullptr;
}

bool
AbstractParameter::hasMinValue() const
{
    return mMinValue != nullptr;
}

bool
AbstractParameter::hasMaxValue() const
{
    return mMaxValue != nullptr;
}

bool
AbstractParameter::hasValueOptions() const
{
    return mValueOptions.size() > 0;
}

AbstractValue&
AbstractParameter::value()
{
    if(mValue == nullptr) throw dab::Exception("DATA ERROR: control parameter has not value", __FILE__, __FUNCTION__, __LINE__);
    
    return *mValue;
}

const AbstractValue&
AbstractParameter::value() const
{
    if(mValue == nullptr) throw dab::Exception("DATA ERROR: control parameter has not value", __FILE__, __FUNCTION__, __LINE__);
    
    return *mValue;
}


void
AbstractParameter::registerParameterListener( std::shared_ptr<ParameterListener> pParameterListener )
{
    // check if parameter listener is already registered
    for(auto iter = mParameterListeners.begin(); iter != mParameterListeners.end(); ++iter)
    {
        std::shared_ptr<ParameterListener> storedListener = iter->lock();
        
        if(storedListener != nullptr && storedListener == pParameterListener) return;
    }
    
    std::weak_ptr<ParameterListener> newListener = pParameterListener;
    
    mParameterListeners.push_back(newListener);
}

void
AbstractParameter::removeParameterListener( std::shared_ptr<ParameterListener> pParameterListener )
{
    // check if parameter listener is registered
    for(auto iter = mParameterListeners.begin(); iter != mParameterListeners.end(); ++iter)
    {
        std::shared_ptr<ParameterListener> storedListener = iter->lock();
        
        if(storedListener != nullptr && storedListener == pParameterListener)
        {
            mParameterListeners.erase(iter);
            return;
        }
    }
}

void
AbstractParameter::updateParameterListeners()
{
    // remove inexisting listeneres
    auto iter = mParameterListeners.end();
    while( iter != mParameterListeners.begin() )
    {
        iter--;
        
        std::shared_ptr<ParameterListener> listener = iter->lock();
        if(listener == nullptr) mParameterListeners.erase(iter);
    }
    
    // notify listeners
    iter = mParameterListeners.begin();
    while(iter != mParameterListeners.end())
    {
        std::shared_ptr<ParameterListener> listener = iter->lock();        
        listener->notifyParameterChange(*this);
        
        iter++;
    }
}

void
AbstractParameter::setName( const std::string& pName )
{
    mName = pName;
}

#pragma mark Parameter implementation

template <>
void
Parameter<std::string>::setValue( const std::string& pValue )
{
    if( mValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no value", __FILE__,__FUNCTION__, __LINE__);
    
    int optionCount = mValueOptions.size();
    
    try
    {
        if( optionCount > 0 )
        {
            bool optionFound = false;
            for(int oI=0; oI<optionCount; ++oI)
            {
                if(mValueOptions[oI]->template get<std::string>() == pValue)
                {
                    optionFound = true;
                    break;
                }
            }
            
            if(optionFound == false) throw dab::Exception("DATA ERROR: parameter has no value option " + pValue, __FILE__,__FUNCTION__, __LINE__);
        }
        
        mValue->set<std::string>(pValue);
        
        updateParameterListeners();
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

#pragma mark Array Parameter implementation

//template <>
//void
//Parameter<std::string>::setValue( const std::string& pValue ) throw (dab::Exception)
//{
//    if( mValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no value", __FILE__,__FUNCTION__, __LINE__);
//    
//    int optionCount = mValueOptions.size();
//    
//    try
//    {
//        if( optionCount > 0 )
//        {
//            bool optionFound = false;
//            for(int oI=0; oI<optionCount; ++oI)
//            {
//                if(mValueOptions[oI]->template get<std::string>() == pValue)
//                {
//                    optionFound = true;
//                    break;
//                }
//            }
//            
//            if(optionFound == false) throw dab::Exception("DATA ERROR: parameter has no value option " + pValue, __FILE__,__FUNCTION__, __LINE__);
//        }
//        
//        mValue->set<std::string>(pValue);
//    }
//    catch(dab::Exception& e)
//    {
//        throw;
//    }
//}




/** \file dab_parameter.h
*/

#pragma once

#include <iostream>
#include <memory>
#include "dab_value.h"
#include "dab_exception.h"

namespace dab
{
    
class AbstractParameter;
    
#pragma mark ParameterListener
    
class ParameterListener
{
public:
    virtual void notifyParameterChange( const AbstractParameter& pParameter ) = 0;
};
    
#pragma mark AbstractParameter definition

class AbstractParameter
{
public:
    AbstractParameter(unsigned int pDataDim = 1);
    AbstractParameter(const std::string& pName, unsigned int pDataDim = 1);
    AbstractParameter(const AbstractParameter& pParameter);
    virtual ~AbstractParameter();
    
    virtual const AbstractParameter& operator=( const AbstractParameter& pParameter ) = 0;
    virtual AbstractParameter* copy() const = 0;
    
    const std::string name() const;

    unsigned int dataDim() const;
    template<typename DataType> bool hasDataType() const;
    bool hasValue() const;
    bool hasMinValue() const;
    bool hasMaxValue() const;
    bool hasValueOptions() const;
    
    AbstractValue& value();
    const AbstractValue& value() const;
    
    template<typename DataType> DataType& value();
    template<typename DataType> DataType& minValue();
    template<typename DataType> DataType& maxValue();
    template<typename DataType> std::vector< DataType& > valueOptions();
    
    template<typename DataType> const DataType& value() const;
    template<typename DataType> const DataType& minValue() const;
    template<typename DataType> const DataType& maxValue() const;
    template<typename DataType> std::vector< const DataType& > valueOptions() const;
    
    void setName( const std::string& pName );
    template<typename DataType > void setValue( const DataType& pValue );
    template<typename DataType > void setMinValue( const DataType& pMinValue );
    template<typename DataType > void setMaxValue( const DataType& pMaxValue );
    
    void registerParameterListener( std::shared_ptr<ParameterListener> pParameterListener );
    void removeParameterListener( std::shared_ptr<ParameterListener> pParameterListener );
    
    virtual operator std::string() const = 0;
    
    friend std::ostream& operator<< (std::ostream & pOstream, const AbstractParameter& pParameter)
    {
        pOstream << std::string( pParameter );
        return pOstream;
    }
    
protected:
    std::string mName;
    int mDataDim;
    AbstractValue* mValue;
    AbstractValue* mMinValue;
    AbstractValue* mMaxValue;
    std::vector<AbstractValue*> mValueOptions;
    std::vector< std::weak_ptr<ParameterListener> > mParameterListeners;
    
    virtual void updateParameterListeners();
};
    
#pragma mark Parameter definition
    
template <typename DataType>
class Parameter : public AbstractParameter
{
public:
    Parameter();
    Parameter(const std::string& pName);
    Parameter(const std::string& pName, const DataType& pValue);
    Parameter(const std::string& pName, const DataType& pValue, const DataType& pMinValue, const DataType& pMaxValue);
    Parameter(const std::string& pName, const DataType& pValue, const std::vector<DataType>& pOptions);
    Parameter(const Parameter& pParameter);
    ~Parameter();
    
    const AbstractParameter& operator=( const AbstractParameter& pParameter );
    const Parameter& operator=( const Parameter& pParameter );
    AbstractParameter* copy() const;
    
    DataType& value();
    DataType& minValue();
    DataType& maxValue();
    std::vector< DataType& > valueOptions();
    
    const DataType& value() const;
    const DataType& minValue() const;
    const DataType& maxValue() const;
    std::vector< const DataType& > valueOptions() const;
    
    void setValue( const DataType& pValue );
    void setMinValue( const DataType& pMinValue );
    void setMaxValue( const DataType& pMaxValue );
    
    operator std::string() const;
    
    friend std::ostream& operator<< (std::ostream & pOstream, const Parameter& pParameter)
    {
        pOstream << std::string( pParameter );
        return pOstream;
    }
    
protected:
};

#pragma mark Array Parameter definition
    
template <typename DataType, int DataDim>
class Parameter< std::array<DataType, DataDim> > : public AbstractParameter
{
public:
    Parameter();
    Parameter(const std::string& pName);
    Parameter(const std::string& pName, const std::array<DataType, DataDim>& pValue);
    Parameter(const std::string& pName, const std::array<DataType, DataDim>& pValue, const std::array<DataType, DataDim>& pMinValue, const std::array<DataType, DataDim>& pMaxValue);
    Parameter(const std::string& pName, const std::array<DataType, DataDim>& pValue, const std::vector< std::array<DataType, DataDim> >& pOptions);
    Parameter(const Parameter& pParameter);
    ~Parameter();
    
    const AbstractParameter& operator=( const AbstractParameter& pParameter );
    const Parameter& operator=( const Parameter& pParameter );
    AbstractParameter* copy() const;
    
    std::array<DataType, DataDim>& value();
    std::array<DataType, DataDim>& minValue();
    std::array<DataType, DataDim>& maxValue();
    std::vector< std::array<DataType, DataDim>& > valueOptions() ;
    
    const std::array<DataType, DataDim>& value() const;
    const std::array<DataType, DataDim>& minValue() const;
    const std::array<DataType, DataDim>& maxValue() const;
    std::vector< const std::array<DataType, DataDim>& > valueOptions() const;
    
    void setValue( const std::array<DataType, DataDim>& pValue );
    void setMinValue( const std::array<DataType, DataDim>& pMinValue );
    void setMaxValue( const std::array<DataType, DataDim>& pMaxValue );
    
    operator std::string() const;
    
    friend std::ostream& operator<< (std::ostream & pOstream, const Parameter& pParameter)
    {
        pOstream << std::string( pParameter );
        return pOstream;
    }
    
protected:
};

    
#pragma mark AbstractParameter implementation
    
template<typename DataType>
bool
AbstractParameter::hasDataType() const
{
    return mValue->typeMatch<DataType>();
}

template<typename DataType>
DataType&
AbstractParameter::value()
{
    if( mValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        return mValue->get<DataType>();
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template<typename DataType>
DataType&
AbstractParameter::minValue()
{
    if( mMinValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no minimum value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        return mMinValue->get<DataType>();
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template<typename DataType>
DataType&
AbstractParameter::maxValue()
{
    if( mMaxValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no maximum value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        return mMaxValue->get<DataType>();
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template<typename DataType>
std::vector< DataType& >
AbstractParameter::valueOptions()
{
    if( mValueOptions.size() == 0 ) throw dab::Exception("DATA ERROR: parameter has no value options", __FILE__,__FUNCTION__, __LINE__);
    
    std::vector< DataType& > options;
    int optionCount = mValueOptions.size();
    
    try
    {
        for(int oI=0; oI<optionCount; ++oI)
        {
            options.push_back( mValueOptions[oI]->get<DataType>() );
        }
    }
    catch(dab::Exception& e)
    {
        throw;
    }
    
    return options;
}

template<typename DataType>
const DataType&
AbstractParameter::value() const
{
    if( mValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        return mValue->get<DataType>();
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template<typename DataType>
const DataType&
AbstractParameter::minValue() const
{
    if( mMinValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no minimum value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        return mMinValue->get<DataType>();
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template<typename DataType>
const DataType&
AbstractParameter::maxValue() const
{
    if( mMaxValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no maximum value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        return mMaxValue->get<DataType>();
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template<typename DataType>
std::vector< const DataType& >
AbstractParameter::valueOptions() const
{
    if( mValueOptions.size() == 0 ) throw dab::Exception("DATA ERROR: parameter has no value options", __FILE__,__FUNCTION__, __LINE__);
    
    std::vector< DataType& > options;
    int optionCount = mValueOptions.size();
    
    try
    {
        for(int oI=0; oI<optionCount; ++oI)
        {
            options.push_back( mValueOptions[oI]->get<DataType>() );
        }
    }
    catch(dab::Exception& e)
    {
        throw;
    }
    
    return options;
}

template<typename DataType >
void
AbstractParameter::setValue( const DataType& pValue )
{
    if( mValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no value", __FILE__,__FUNCTION__, __LINE__);
    
    DataType _Value = pValue;
    
    try
    {
        int optionCount = mValueOptions.size();
        
        if( optionCount > 0 )
        {
            bool optionFound = false;
            for(int oI=0; oI<optionCount; ++oI)
            {
                if(mValueOptions[oI]->get<DataType>() == pValue)
                {
                    optionFound = true;
                    break;
                }
            }
            
            if(optionFound == false) throw dab::Exception("DATA ERROR: parameter has no value option ", __FILE__,__FUNCTION__, __LINE__);
        }
        
        if( mMinValue != nullptr )
        {
            if( _Value < mMinValue->get<DataType>() ) _Value = mMinValue->get<DataType>();
        }
        
        if( mMaxValue != nullptr )
        {
            if( _Value > mMaxValue->get<DataType>() ) _Value = mMaxValue->get<DataType>();
        }
        
        mValue->set<DataType>(pValue);
        
        updateParameterListeners();
        
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template<typename DataType >
void
AbstractParameter::setMinValue( const DataType& pMinValue )
{
    if( mMinValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no minimum value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        mMinValue->set<DataType>(pMinValue);
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template<typename DataType >
void
AbstractParameter::setMaxValue( const DataType& pMaxValue )
{
    if( mMaxValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no maximum value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        mMaxValue->set<DataType>(pMaxValue);
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}
    
#pragma mark Parameter implementation
    
template <typename DataType>
Parameter<DataType>::Parameter()
: AbstractParameter()
{}

template <typename DataType>
Parameter<DataType>::Parameter(const std::string& pName)
: AbstractParameter(pName, 1)
{}

template <typename DataType>
Parameter<DataType>::Parameter(const std::string& pName, const DataType& pValue)
: AbstractParameter(pName, 1)
{
    mValue = new Value<DataType>(pValue);
}

template <typename DataType>
Parameter<DataType>::Parameter(const std::string& pName, const DataType& pValue, const DataType& pMinValue, const DataType& pMaxValue)
: AbstractParameter(pName, 1)
{
    mValue = new Value<DataType>(pValue);
    mMinValue = new Value<DataType>(pMinValue);
    mMaxValue = new Value<DataType>(pMaxValue);
}

template <typename DataType>
Parameter<DataType>::Parameter(const std::string& pName, const DataType& pValue, const std::vector<DataType>& pOptions)
: AbstractParameter(pName, 1)
{
    mValue = new Value<DataType>(pValue);
    
    int optionCount = pOptions.size();
    for(int oI=0; oI<optionCount; ++oI)
    {
        mValueOptions.push_back( new Value<DataType>( pOptions[oI] ) );
    }
}

template <typename DataType>
Parameter<DataType>::Parameter(const Parameter<DataType>& pParameter)
: AbstractParameter(pParameter)
{
    mDataDim = pParameter.mDataDim;
    
    if( pParameter.mValue != nullptr ) mValue = pParameter.mValue->copy();
    if( pParameter.mMinValue != nullptr ) mMinValue = pParameter.mMinValue->copy();
    if( pParameter.mMaxValue != nullptr ) mMaxValue = pParameter.mMaxValue->copy();
    if( pParameter.mValueOptions.size() > 0 )
    {
        int optionCount = pParameter.mValueOptions.size();
        for(int oI=0; oI<optionCount; ++oI)
        {
            mValueOptions.push_back( pParameter.mValueOptions[oI]->copy() );
        }
    }
}

template <typename DataType>
Parameter<DataType>::~Parameter()
{
    if( mValue != nullptr ) delete mValue;
    if( mMinValue != nullptr ) delete mMinValue;
    if( mMaxValue != nullptr ) delete mMaxValue;
    if( mValueOptions.size() > 0 )
    {
        int optionCount = mValueOptions.size();
        for(int oI=0; oI<optionCount; ++oI) delete mValueOptions[oI];
        mValueOptions.clear();
    }
}

template <typename DataType>
const AbstractParameter&
Parameter<DataType>::operator=( const AbstractParameter& pParameter )
{
    if( dynamic_cast< const Parameter<DataType>* >(&pParameter) == nullptr ) throw dab::Exception( "DATA ERROR: incompatible data type", __FILE__, __FUNCTION__, __LINE__ );
    
    return operator=( static_cast< const Parameter<DataType>& >(pParameter) );
}

template <typename DataType>
const Parameter<DataType>&
Parameter<DataType>::operator=( const Parameter<DataType>& pParameter )
{
    mDataDim = pParameter.mDataDim;
    
    if( pParameter.mValue != nullptr ) mValue = pParameter.mValue->copy();
    if( pParameter.mMinValue != nullptr ) mMinValue = pParameter.mMinValue->copy();
    if( pParameter.mMaxValue != nullptr ) mMaxValue = pParameter.mMaxValue->copy();
    if( pParameter.mValueOptions.size() > 0 )
    {
        int optionCount = pParameter.mValueOptions.size();
        for(int oI=0; oI<optionCount; ++oI)
        {
            mValueOptions.push_back( pParameter.mValueOptions[oI]->copy());
        }
    }
    
    return *this;
}

template <typename DataType>
AbstractParameter*
Parameter<DataType>::copy() const
{
    return new Parameter<DataType>(*this);
}

template <typename DataType>
DataType&
Parameter<DataType>::value()
{
    if( mValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no value", __FILE__,__FUNCTION__, __LINE__);
    
    return mValue->get<DataType>();
}

template <typename DataType>
DataType&
Parameter<DataType>::minValue()
{
    if( mMinValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no minimum value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        return mMinValue->get<DataType>();
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template <typename DataType>
DataType&
Parameter<DataType>::maxValue()
{
    if( mMaxValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no maximum value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        return mMaxValue->get<DataType>();
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template <typename DataType>
std::vector< DataType& >
Parameter<DataType>::valueOptions()
{
    if( mValueOptions.size() == 0 ) throw dab::Exception("DATA ERROR: parameter has no value options", __FILE__,__FUNCTION__, __LINE__);
    
    int optionCount = mValueOptions.size();
    std::vector< DataType& > options;
    
    try
    {
        for(int oI=0; oI<optionCount; ++oI)
        {
            options.push_back( mValueOptions[oI]->template get<DataType>() );
        }
    }
    catch(dab::Exception& e)
    {
        throw;
    }
    
    return options;
}

template <typename DataType>
const DataType&
Parameter<DataType>::value() const
{
    if( mValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        return mValue->get<DataType>();
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template <typename DataType>
const DataType&
Parameter<DataType>::minValue() const
{
    if( mMinValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no minimum value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        return mMinValue->get<DataType>();
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template <typename DataType>
const DataType&
Parameter<DataType>::maxValue() const
{
    if( mMaxValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no maximum value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        return mMaxValue->get<DataType>();
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template <typename DataType>
std::vector< const DataType& >
Parameter<DataType>::valueOptions() const
{
    if( mValueOptions.size() == 0 ) throw dab::Exception("DATA ERROR: parameter has no value options", __FILE__,__FUNCTION__, __LINE__);
    
    int optionCount = mValueOptions.size();
    std::vector< DataType& > options;
    
    try
    {
        for(int oI=0; oI<optionCount; ++oI)
        {
            options.push_back( mValueOptions[oI]->template get<DataType>() );
        }
    }
    catch(dab::Exception& e)
    {
        throw;
    }
    
    return options;
}

template <>
void
Parameter<std::string>::setValue( const std::string& pValue );

template <typename DataType>
void
Parameter<DataType>::setValue( const DataType& pValue )
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
                if(mValueOptions[oI]->template get<DataType>() == pValue)
                {
                    optionFound = true;
                    break;
                }
            }
            
            if(optionFound == false) throw dab::Exception("DATA ERROR: parameter has no value option " + to_string(pValue), __FILE__,__FUNCTION__, __LINE__);
        }
        
        mValue->set<DataType>(pValue);
        
        updateParameterListeners();
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template <typename DataType>
void
Parameter<DataType>::setMinValue( const DataType& pMinValue )
{
    if( mMinValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no minimum value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        mMinValue->set<DataType>(pMinValue);
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template <typename DataType>
void
Parameter<DataType>::setMaxValue( const DataType& pMaxValue )
{
    if( mMaxValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no maximum value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        mMaxValue->set<DataType>(pMaxValue);
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template <typename DataType>
Parameter<DataType>::operator std::string() const
{
    std::stringstream ss;
    
    ss << "Parameter name: " << mName;
    
//    // debug
//    if( mValue != nullptr ) ss << *mValue << "\n";
//    // debug done
    
    if( mValue != nullptr ) ss << "value: " << mValue->get<DataType>();
    if( mMinValue != nullptr ) ss << " min: " << mMinValue->get<DataType>();
    if( mMaxValue != nullptr ) ss << " min: " << mMaxValue->get<DataType>();
    
    if( mValueOptions.size() > 0 )
    {
        ss << " options:";
        for(int oI=0; oI<mValueOptions.size(); ++oI) ss << " " << mValueOptions[oI]->template get<DataType>();
        ss << "\n";
    }
    
    return ss.str();
}


#pragma mark Array Parameter implementation

template <typename DataType, int DataDim>
Parameter< std::array<DataType, DataDim> >::Parameter()
: AbstractParameter(DataDim)
{}

template <typename DataType, int DataDim>
Parameter< std::array<DataType, DataDim> >::Parameter(const std::string& pName)
: AbstractParameter(pName, DataDim)
{}

template <typename DataType, int DataDim>
Parameter< std::array<DataType, DataDim> >::Parameter(const std::string& pName, const std::array<DataType, DataDim>& pValue)
: AbstractParameter(pName, DataDim)
{
    mValue = new Value< std::array<DataType, DataDim> >(pValue);
}

template <typename DataType, int DataDim>
Parameter< std::array<DataType, DataDim> >::Parameter(const std::string& pName, const std::array<DataType, DataDim>& pValue, const std::array<DataType, DataDim>& pMinValue, const std::array<DataType, DataDim>& pMaxValue)
: AbstractParameter(pName, DataDim)
{
    mValue = new Value< std::array<DataType, DataDim> >(pValue);
    mMinValue = new Value< std::array<DataType, DataDim> >(pMinValue);
    mMaxValue = new Value< std::array<DataType, DataDim> >(pMaxValue);
}

template <typename DataType, int DataDim>
Parameter< std::array<DataType, DataDim> >::Parameter(const std::string& pName, const std::array<DataType, DataDim>& pValue, const std::vector< std::array<DataType, DataDim> >& pOptions)
: AbstractParameter(pName, DataDim)
{
    mValue = new Value< std::array<DataType, DataDim> >(pValue);
    
    int optionCount = pOptions.size();
    for(int oI=0; oI<optionCount; ++oI)
    {
        mValueOptions.push_back( new Value< std::array<DataType, DataDim> >( pOptions[oI] ) );
    }
}

template <typename DataType, int DataDim>
Parameter< std::array<DataType, DataDim> >::Parameter(const Parameter< std::array<DataType, DataDim> >& pParameter)
: AbstractParameter(pParameter, DataDim)
{
    if( pParameter.mValue != nullptr ) mValue = pParameter.mValue->copy();
    if( pParameter.mMinValue != nullptr ) mMinValue = pParameter.mMinValue->copy();
    if( pParameter.mMaxValue != nullptr ) mMaxValue = pParameter.mMaxValue->copy();
    if( pParameter.mValueOptions.size() > 0 )
    {
        int optionCount = pParameter.mValueOptions.size();
        for(int oI=0; oI<optionCount; ++oI)
        {
            mValueOptions.push_back( pParameter.mValueOptions[oI]->copy() );
        }
    }
}

template <typename DataType, int DataDim>
Parameter< std::array<DataType, DataDim> >::~Parameter()
{
    if( mValue != nullptr ) delete mValue;
    if( mMinValue != nullptr ) delete mMinValue;
    if( mMaxValue != nullptr ) delete mMaxValue;
    if( mValueOptions.size() > 0 )
    {
        int optionCount = mValueOptions.size();
        for(int oI=0; oI<optionCount; ++oI) delete mValueOptions[oI];
        mValueOptions.clear();
    }
}

template <typename DataType, int DataDim>
const AbstractParameter&
Parameter< std::array<DataType, DataDim> >::operator=( const AbstractParameter& pParameter )
{
    if( dynamic_cast< const Parameter<DataType>* >(&pParameter) == nullptr ) throw dab::Exception( "DATA ERROR: incompatible data type", __FILE__, __FUNCTION__, __LINE__ );
    
    return operator=( static_cast< const Parameter< std::array<DataType, DataDim> >& >(pParameter) );
}

template <typename DataType, int DataDim>
const Parameter< std::array<DataType, DataDim> >&
Parameter< std::array<DataType, DataDim> >::operator=( const Parameter< std::array<DataType, DataDim> >& pParameter )
{
    mDataDim = pParameter.mDataDim;
    
    if( pParameter.mValue != nullptr ) mValue = pParameter.mValue->copy();
    if( pParameter.mMinValue != nullptr ) mMinValue = pParameter.mMinValue->copy();
    if( pParameter.mMaxValue != nullptr ) mMaxValue = pParameter.mMaxValue->copy();
    if( pParameter.mValueOptions.size() > 0 )
    {
        int optionCount = pParameter.mValueOptions.size();
        for(int oI=0; oI<optionCount; ++oI)
        {
            mValueOptions.push_back( pParameter.mValueOptions[oI]->copy());
        }
    }
    
    return *this;
}

template <typename DataType, int DataDim>
AbstractParameter*
Parameter< std::array<DataType, DataDim> >::copy() const
{
    return new Parameter< std::array<DataType, DataDim> >(*this);
}

template <typename DataType, int DataDim>
std::array<DataType, DataDim>&
Parameter< std::array<DataType, DataDim> >::value()
{
    if( mValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no value", __FILE__,__FUNCTION__, __LINE__);
    
    return mValue->get< std::array<DataType, DataDim> >();
}

template <typename DataType, int DataDim>
std::array<DataType, DataDim>&
Parameter< std::array<DataType, DataDim> >::minValue()
{
    if( mMinValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no minimum value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        return mMinValue->get< std::array<DataType, DataDim> >();
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template <typename DataType, int DataDim>
std::array<DataType, DataDim>&
Parameter< std::array<DataType, DataDim> >::maxValue()
{
    if( mMaxValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no maximum value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        return mMaxValue->get< std::array<DataType, DataDim> >();
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template <typename DataType, int DataDim>
std::vector< std::array<DataType, DataDim>& >
Parameter< std::array<DataType, DataDim> >::valueOptions()
{
    if( mValueOptions.size() == 0 ) throw dab::Exception("DATA ERROR: parameter has no value options", __FILE__,__FUNCTION__, __LINE__);
    
    int optionCount = mValueOptions.size();
    std::vector< DataType& > options;
    
    try
    {
        for(int oI=0; oI<optionCount; ++oI)
        {
            options.push_back( mValueOptions[oI]->template get< std::array<DataType, DataDim> >() );
        }
    }
    catch(dab::Exception& e)
    {
        throw;
    }
    
    return options;
}

template <typename DataType, int DataDim>
const std::array<DataType, DataDim>&
Parameter< std::array<DataType, DataDim> >::value() const
{
    if( mValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        return mValue->get< std::array<DataType, DataDim> >();
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template <typename DataType, int DataDim>
const std::array<DataType, DataDim>&
Parameter< std::array<DataType, DataDim> >::minValue() const
{
    if( mMinValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no minimum value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        return mMinValue->get< std::array<DataType, DataDim> >();
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template <typename DataType, int DataDim>
const std::array<DataType, DataDim>&
Parameter< std::array<DataType, DataDim> >::maxValue() const
{
    if( mMaxValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no maximum value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        return mMaxValue->get< std::array<DataType, DataDim> >();
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template <typename DataType, int DataDim>
std::vector< const std::array<DataType, DataDim>& >
Parameter< std::array<DataType, DataDim> >::valueOptions() const
{
    if( mValueOptions.size() == 0 ) throw dab::Exception("DATA ERROR: parameter has no value options", __FILE__,__FUNCTION__, __LINE__);
    
    int optionCount = mValueOptions.size();
    std::vector< DataType& > options;
    
    try
    {
        for(int oI=0; oI<optionCount; ++oI)
        {
            options.push_back( mValueOptions[oI]->template get< std::array<DataType, DataDim> >() );
        }
    }
    catch(dab::Exception& e)
    {
        throw;
    }
    
    return options;
}

//template <int DataDim>
//void
//Parameter< std::array<std::string, DataDim> >::setValue( const std::array<std::string, DataDim>& pValue );

template <typename DataType, int DataDim>
void
Parameter< std::array<DataType, DataDim> >::setValue( const std::array<DataType, DataDim>& pValue )
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
                if(mValueOptions[oI]->template get< std::array<DataType, DataDim> >() == pValue)
                {
                    optionFound = true;
                    break;
                }
            }
            
            if(optionFound == false) throw dab::Exception("DATA ERROR: parameter has no value option " + to_string(pValue), __FILE__,__FUNCTION__, __LINE__);
        }
        
        
        if( mMinValue != nullptr )
        {
            const std::array<DataType, DataDim>& minValues = mMinValue->get< std::array<DataType, DataDim> >();
            
            for(int d=0; d<DataDim; ++d)
            {
                if( pValue[d] < minValues[d] ) pValue[d] = minValues[d];
            }
        }
        
        if( mMaxValue != nullptr )
        {
            const std::array<DataType, DataDim>& maxValues = mMaxValue->get< std::array<DataType, DataDim> >();
            
            for(int d=0; d<DataDim; ++d)
            {
                if( pValue[d] > maxValues[d] ) pValue[d] = maxValues[d];
            }
        }
        
        mValue->set< std::array<DataType, DataDim> >(pValue);
        
        updateParameterListeners();
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template <typename DataType, int DataDim>
void
Parameter< std::array<DataType, DataDim> >::setMinValue( const std::array<DataType, DataDim>& pMinValue )
{
    if( mMinValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no minimum value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        mMinValue->set< std::array<DataType, DataDim> >(pMinValue);
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template <typename DataType, int DataDim>
void
Parameter< std::array<DataType, DataDim> >::setMaxValue( const std::array<DataType, DataDim>& pMaxValue )
{
    if( mMaxValue == nullptr ) throw dab::Exception("DATA ERROR: parameter has no maximum value", __FILE__,__FUNCTION__, __LINE__);
    
    try
    {
        mMaxValue->set< std::array<DataType, DataDim> >(pMaxValue);
    }
    catch(dab::Exception& e)
    {
        throw;
    }
}

template <typename DataType, int DataDim>
Parameter< std::array<DataType, DataDim> >::operator std::string() const
{
    std::stringstream ss;
    
    ss << "Parameter name: " << mName;
    
    if( mValue != nullptr ) ss << " value:";
    for(int d=0; d<DataDim; ++d) ss << " " << mValue->get< std::array<DataType, DataDim> >()[d];
    if( mMinValue != nullptr ) ss << " min:";
    for(int d=0; d<DataDim; ++d) ss << " " << mMinValue->get< std::array<DataType, DataDim> >()[d];
    if( mMaxValue != nullptr ) ss << " min:";
    for(int d=0; d<DataDim; ++d) ss << " " << mMaxValue->get< std::array<DataType, DataDim> >()[d];
    
    if( mValueOptions.size() > 0 )
    {
        ss << " options:";
        for(int oI=0; oI<mValueOptions.size(); ++oI)
        {
            ss << " [";
            
            for(int d=0; d<DataDim; ++d)
            {
                ss << " " << mValueOptions[oI]->template get< std::array<DataType, DataDim> >()[d];
            }

            ss << " ]";
        }
        ss << "\n";
    }
    
    return ss.str();
}


};

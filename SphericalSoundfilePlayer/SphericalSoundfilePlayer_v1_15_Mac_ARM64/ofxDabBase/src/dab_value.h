/** \file dab_value.h
*/

#pragma once

#include <iostream>
#include <sstream>
#include <typeindex>
#include "dab_exception.h"

namespace dab
{
#pragma mark AbstractValue definition
    
class AbstractValue
{
public:
    template <class Type> AbstractValue();
    template <class Type> AbstractValue( const Type& pValue );
    virtual ~AbstractValue();
    
    virtual const AbstractValue& operator=( const AbstractValue& pValue ) = 0;
    virtual AbstractValue* copy() const = 0;
    
    const std::type_index& type() const;
    virtual unsigned int dim() const = 0;
    template <class Type> inline bool typeMatch() const;
    bool typeMatch( const AbstractValue& pValue ) const;
    
    template <class Type> operator Type() const;
    template <class Type> void get( Type& pValue ) const;
    template <class Type> Type& get();
    template <class Type> const Type& get() const;
    
    template <class Type> void set(const Type& pType) const;
    
    virtual std::string info() const;
    
    friend std::ostream& operator<< (std::ostream & pOstream, const AbstractValue& pValue)
    {
        pOstream << pValue.info();
        
        return pOstream;
    }       
    
protected:
    void* mValue;
    std::type_index mType;
};
    
#pragma mark Value definition
    
template <class Type>
class Value : public AbstractValue
{
public:
    Value();
    Value( Type pValue );
    Value( const Value& pValue );
    ~Value();
    
    const AbstractValue& operator=( const AbstractValue& pValue );
    const Value& operator=( const Value& pValue );
    template <class Type2> const Value& operator=( const Value<Type2>& pValue );
    AbstractValue* copy() const;
    
    Type& get();
    const Type& get() const;
    unsigned int dim() const;
    operator Type();
    
    void set(const Type& pValue);
    
    std::string info() const;
    
    friend std::ostream& operator<< (std::ostream & pOstream, const Value& pValue)
    {
        pOstream << pValue.info();
        
        return pOstream;
    }       
};
    
template< class Type, int Dim >
class Value< std::array<Type,Dim> > : public AbstractValue
{
public:
    Value();
    Value( const std::array<Type,Dim>& pValue );
    Value( const Value& pValue );
    ~Value();
    
    const AbstractValue& operator=( const AbstractValue& pValue );
    const Value& operator=( const Value& pValue );
    AbstractValue* copy() const;
    
    std::array<Type, Dim>& get();
    const std::array<Type, Dim>& get() const;
    unsigned int dim() const;
    operator std::array<Type, Dim>();
    
    void set(const std::array<Type, Dim>& pValue);

    std::string info() const;
};
    
#pragma mark AbstractValue implementation
    
template <class Type>
AbstractValue::AbstractValue( )
{
    Type* type = new Type();
    mType = std::type_index(typeid(Type));
    mValue = static_cast<void*>( type );
}

template <class Type>
AbstractValue::AbstractValue( const Type& pValue )
: mType( std::type_index(typeid(Type)) )
{
    mValue = static_cast<void*>( new Type( pValue ) );
}
    
template <class Type>
bool
AbstractValue::typeMatch() const
{
    return std::type_index(typeid(Type) ) == mType;
}
    
template <class Type>
AbstractValue::operator Type() const
{
    if( typeMatch<Type>() == false ) throw( dab::Exception( "TYPE ERROR: supplied type " + std::string(std::type_index(typeid(Type)).name()) + " doesn't match stored type " + std::string(mType.name()), __FILE__, __FUNCTION__, __LINE__ ));
    
    return *( static_cast<Type*>( mValue ) );
}

template <class Type>
void
AbstractValue::get( Type& pValue ) const
{
    if( typeMatch<Type>() == false ) throw( dab::Exception( "TYPE ERROR: supplied type " + std::string(std::type_index(typeid(Type)).name()) + " doesn't match stored type " + std::string(mType.name()), __FILE__, __FUNCTION__, __LINE__ ));
    
    pValue = *( static_cast<Type*>( mValue ) );
}

template <class Type>
Type&
AbstractValue::get()
{
    if( typeMatch<Type>() == false ) throw( dab::Exception( "TYPE ERROR: supplied type " + std::string(std::type_index(typeid(Type)).name()) + " doesn't match stored type " + std::string(mType.name()), __FILE__, __FUNCTION__, __LINE__ ));
    
    return *( static_cast<Type*>( mValue ) );
}

template <class Type>
const Type&
AbstractValue::get() const
{
    if( typeMatch<Type>() == false ) throw( dab::Exception( "TYPE ERROR: supplied type " + std::string(std::type_index(typeid(Type)).name()) + " doesn't match stored type " + std::string(mType.name()), __FILE__, __FUNCTION__, __LINE__ ));
    
    return *( static_cast<Type*>( mValue ) );
}
    
template <class Type>
void
AbstractValue::set(const Type& pType) const
{
    if( typeMatch<Type>() == false ) throw( dab::Exception( "TYPE ERROR: supplied type " + std::string(std::type_index(typeid(Type)).name()) + " doesn't match stored type " + std::string(mType.name()), __FILE__, __FUNCTION__, __LINE__ ));
    
    *(static_cast<Type*>(mValue)) = pType;
}
    
#pragma mark Value implementation
    
template <class Type>
Value<Type>::Value( )
: AbstractValue( Type() )
{}
    
template <class Type>
Value<Type>::Value( Type pValue )
: AbstractValue( pValue )
{}
    
template <class Type>
Value<Type>::Value( const Value& pValue )
: AbstractValue( *( static_cast<Type*>( pValue.mValue ) ) )
{}
    
template <class Type>
Value<Type>::~Value()
{
    delete static_cast<Type*>( mValue );
}
  
template <class Type>
const AbstractValue&
Value<Type>::operator=( const AbstractValue& pValue )
{
    if( pValue.typeMatch<Type>() == false ) throw dab::Exception("TYPE ERROR: supplied type " + std::string(pValue.type().name()) + " doesn't match stored type " + std::string(mType.name()), __FILE__, __FUNCTION__, __LINE__ );

    try
    {
        Type& mTypedValue = this->get();
        
        const Type& pTypedValue = pValue.get<Type>();
        
        mTypedValue = pTypedValue;

        return *this;
    }
    catch(dab::Exception& e)
    {
        e += dab::Exception("TYPE ERROR: assignment operator failed\n", __FILE__, __FUNCTION__, __LINE__ );
        throw e;
    }
}
 
template <class Type>
const Value<Type>&
Value<Type>::operator=( const Value<Type>& pValue )
{
    Type& mTypedValue = this->get();
    
    const Type& pTypedValue = pValue.get();
    
    mTypedValue = pTypedValue;
    
    return *this;
}
    
template <class Type>
template <class Type2>
const Value<Type>&
Value<Type>::operator=( const Value<Type2>& pValue )
{
    *( static_cast<Type*>( mValue ) ) = pValue.get();
    
    return *this;
}
    
template <class Type>
unsigned int
Value<Type>::dim() const
{
    return 1;
}
    
template <class Type>
Type&
Value<Type>::get()
{
    return *( static_cast<Type*>( mValue ) );
}

template <class Type>
const Type&
Value<Type>::get() const
{
    return *( static_cast<Type*>( mValue ) );
}
    
template <class Type>
Value<Type>::operator Type()
{
    return *( static_cast<Type*>( mValue ) );
}
 
template <class Type>
void
Value<Type>::set(const Type& pValue)
{
    *(static_cast<Type*>( mValue )) = pValue;
}
    
template <class Type>
AbstractValue*
Value<Type>::copy() const
{
    return new Value( *this );
}
    
template <class Type>
std::string
Value<Type>::info() const
{
    std::stringstream ss;
    
    ss << "value type " << mType.name() << " value " << *(static_cast<Type*>(mValue));

    return ss.str();
}

template< class Type, int Dim >
Value< std::array<Type,Dim> >::Value()
    : AbstractValue( std::array<Type,Dim>() )
{}
  
template< class Type, int Dim >
Value< std::array<Type,Dim> >::Value( const std::array<Type,Dim>& pValue )
    : AbstractValue( pValue )
{}
    
template< class Type, int Dim >
Value< std::array<Type,Dim> >::Value( const Value& pValue )
    : AbstractValue( *( static_cast< std::array<Type,Dim>* >( pValue.mValue ) ) )
{}
    
template< class Type, int Dim >
Value< std::array<Type,Dim> >::~Value()
{
    delete static_cast< std::array<Type,Dim>* >( mValue );
}
  
template< class Type, int Dim >
const AbstractValue&
Value< std::array<Type,Dim> >::operator=( const AbstractValue& pValue )
{
    if( pValue.typeMatch< std::array<Type,Dim> >() == false ) throw dab::Exception("TYPE ERROR: supplied type " + std::string(pValue.type().name()) + " doesn't match stored type " + std::string(mType.name()), __FILE__, __FUNCTION__, __LINE__ );
    
    try
    {
        std::array<Type, Dim>& mTypedValue = this->get();
        
        const std::array<Type, Dim>& pTypedValue = pValue.get< std::array<Type, Dim> >();
        
        mTypedValue = pTypedValue;
        
        return *this;
    }
    catch(dab::Exception& e)
    {
        e += dab::Exception("TYPE ERROR: assignment operator failed\n", __FILE__, __FUNCTION__, __LINE__ );
    }
}
    
template< class Type, int Dim >
const Value< std::array<Type,Dim> >&
Value< std::array<Type,Dim> >::operator=( const Value< std::array<Type,Dim>  >& pValue )
{
    Type& mTypedValue = this->get();
    
    const Type& pTypedValue = pValue.get();
    
    mTypedValue = pTypedValue;
    
    return *this;
}

template< class Type, int Dim >
std::array<Type, Dim>&
Value< std::array<Type, Dim> >::get()
{
    return *( static_cast< std::array<Type, Dim>* >( mValue ) );
}

template< class Type, int Dim >
const std::array<Type, Dim>&
Value< std::array<Type, Dim> >::get() const
{
    return *( static_cast< std::array<Type, Dim>* >( mValue ) );
}

template< class Type, int Dim >
unsigned int
Value< std::array<Type, Dim> >::dim() const
{
    return static_cast< std::array<Type, Dim>* >( mValue )->size();
}

template< class Type, int Dim >
Value< std::array<Type, Dim> >::operator std::array<Type, Dim>()
{
    return *( static_cast< std::array<Type, Dim>* >( mValue ) );
}

template< class Type, int Dim >
void
Value< std::array<Type, Dim> >::set(const std::array<Type, Dim>& pValue)
{
    *( static_cast< std::array<Type, Dim>* >( mValue ) ) = pValue;
}

template< class Type, int Dim >
AbstractValue*
Value< std::array<Type,Dim> >::copy() const
{
    return new Value( *this );
}

template< class Type, int Dim >
std::string
Value< std::array<Type,Dim> >::info() const
{
    std::stringstream ss;
    
    ss << "value type " << mType.name() << " value ";
    
    std::array<Type, Dim>* array = static_cast< std::array<Type, Dim>* >( mValue );
    
    for(int d=0; d<Dim; ++d) ss << (*array)[d] << " ";
    std::cout << "\n";
    
    return ss.str();
}
    
};

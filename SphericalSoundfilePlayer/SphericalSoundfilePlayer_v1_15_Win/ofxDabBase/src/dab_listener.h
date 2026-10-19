/** \file dab_listener.h
*/

#pragma once

#include <vector>
#include <memory>
#include "dab_value.h"

namespace dab
{
  
class UpdateListener;
    
#pragma mark UpdateNotifier definition
    
class UpdateNotifier
{
public:
    void addListener( std::shared_ptr<UpdateListener> pListener );
    void removeListener( std::shared_ptr<UpdateListener> pListener );
    void removeListeners();
    virtual void update();
    
protected:
    std::vector< std::weak_ptr<UpdateListener> > mListeners;
};
    
#pragma mark UpdateListener definition

class UpdateListener
{
public:
    virtual ~UpdateListener();
    virtual void notifyUpdate();
    virtual void notifyUpdate(const AbstractValue& pValue);
    
protected:
};
    
//template<typename ListenerType, typename DataType>
//class UpdateListener
//{
//public:
//    void notifyDataUpdate(const DataType& pDataUpdate);
//    
//protected:
//};
    
};

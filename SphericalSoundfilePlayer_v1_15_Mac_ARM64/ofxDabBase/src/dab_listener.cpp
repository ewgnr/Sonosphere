/** \file dab_listener.cpp
*/

#include "dab_listener.h"

using namespace dab;

#pragma mark UpdateNotifier implementation

void
UpdateNotifier::addListener( std::shared_ptr<UpdateListener> pListener )
{
    // check if listener is stored
    int listenerCount = mListeners.size();
    for(int lI=0; lI<listenerCount; ++lI)
    {
       if(mListeners[lI].expired() == false)
       {
           auto listener = mListeners[lI].lock();
           if(listener == pListener) return;
       }
    }
    
    std::weak_ptr<UpdateListener> wP = pListener;
    mListeners.push_back(wP);
}

void
UpdateNotifier::removeListener( std::shared_ptr<UpdateListener> pListener )
{
    // check if listener is stored
    int foundListenerIndex = -1;
    int listenerCount = mListeners.size();
    for(int lI=0; lI<listenerCount; ++lI)
    {
        if(mListeners[lI].expired() == false)
        {
            auto listener = mListeners[lI].lock();
            if(listener == pListener)
            {
                foundListenerIndex = lI;
                break;
            }
        }
    }
    if(foundListenerIndex < 0) return;
    
    mListeners.erase(mListeners.begin() + foundListenerIndex);
}

void
UpdateNotifier::removeListeners()
{
    mListeners.clear();
}

void
UpdateNotifier::update()
{
    // remove expired listeners
    int listenerCount = mListeners.size();
    for(int lI=listenerCount-1; lI>=0; --lI)
    {
        if(mListeners[lI].expired() == true) mListeners.erase(mListeners.begin() + lI);
    }
}

#pragma mark UpdateListener implementation

UpdateListener::~UpdateListener()
{}

void
UpdateListener::notifyUpdate()
{}

void
UpdateListener::notifyUpdate(const AbstractValue& pValue)
{}

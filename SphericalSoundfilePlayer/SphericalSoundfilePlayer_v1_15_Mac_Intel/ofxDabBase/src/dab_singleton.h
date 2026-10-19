/** \file dab_singleton.h
*/

#pragma once

#include <iostream>

namespace dab
{
	
template <typename Type>
class Singleton
{
    
public:
    static Type& get();
    static void destroy();
    
protected:
    Singleton();
    ~Singleton();
    
    static Type* sInstance;
};

template <typename Type> Type* Singleton<Type>::sInstance = NULL;

template <typename Type>
Singleton<Type>::Singleton()
{}

template <typename Type>
Singleton<Type>::~Singleton()
{}

template <typename Type>
Type&
Singleton<Type>::get()
{
    if(sInstance == NULL) sInstance = new Type();
    return *sInstance;
}

template <typename Type>
void
Singleton<Type>::destroy()
{
    if(sInstance != NULL)
    {
        delete sInstance;
        sInstance = NULL;
    }
}

};
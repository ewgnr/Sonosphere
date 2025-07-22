/** \file dab_exception.h
*/

#pragma once

#include <iostream>
#include <vector>
#include <string>

namespace dab
{

class Exception
{
public:
    Exception( const std::string& pFile, const std::string& pFunction, int pLine );
    Exception( const std::string& pMessage, const std::string& pFile, const std::string& pFunction, int pLine );
    Exception( const Exception& pException );
    ~Exception();
    
    Exception& operator=( const Exception& pException );
    Exception& operator+= ( const Exception& pException );
    
    operator std::string() const;
    
    friend std::ostream& operator<< ( std::ostream& pOstream, const Exception& pException )
    {
        pOstream << std::string( pException );
        return pOstream;
    }
    
protected:
    std::vector<std::string> mMessages;
    std::vector< std::string > mFiles;
    std::vector< std::string > mFunctions;
    std::vector< int > mLines;
};
    
};
/** \file dab_tokenizer.h
*/

#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include "dab_singleton.h"

namespace dab
{
        
class Tokenizer : public Singleton<Tokenizer>
{
public:
    friend class Singleton<Tokenizer>;
    
    void parse( const std::string& pString, std::tm& pDate, char pDateDelim = '.', char pTimeDelim = ':', char pDateTimeDelim = ' ' );
    void parse( const std::tm& pDate, std::string& pString , char pDateDelim = '.', char pTimeDelim = ':', char pDateTimeDelim = ' ' );
    template<typename Type> void parse( const std::string& pString, Type& pValue );
    template<typename Type> void parse( const std::string& pString, std::vector<Type>& pValues, char pDelim = ' ' );
    template<typename Type> void parse( const std::vector<Type>& pValues, std::string& pString, char pDelim = ' ' );
    
    void substring( const std::string& pString, std::string& pResult, const std::string& pFrom, const std::string& pTo = "" );
    void split( const std::string& pString, std::vector< std::string >& pStrings, char pDelim );
    
protected:
    Tokenizer();
    ~Tokenizer();
};

template<typename Type>
void
Tokenizer::parse( const std::string& pString, Type& pValue )
{
    std::stringstream ss( pString );
    std::string s;
    ss << s;
    ss >> pValue;
}

template<typename Type>
void
Tokenizer::parse( const std::string& pString, std::vector<Type>& pValues, char pDelim )
{
    std::istringstream iss( pString );
    std::string s;
    Type v;
    
    pValues.clear();
    
    while( getline(iss, s, pDelim) )
    {
        std::stringstream ss;
        ss << s;
        ss >> v;
        pValues.push_back(v);
    }
}

template<typename Type>
void
Tokenizer::parse( const std::vector<Type>& pValues, std::string& pString, char pDelim )
{
    std::stringstream sstream;
    size_t valueCount = pValues.size();
    
    for(size_t i=0; i<valueCount; ++i)
    {
        sstream << pValues[i] << pDelim;
    }
    
    pString = sstream.str();
}
    
};
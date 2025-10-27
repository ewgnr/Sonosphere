/** \file dab_tokenizer.cpp
*/

#include "dab_tokenizer.h"
#include <iomanip>

using namespace dab;

Tokenizer::Tokenizer()
{}

Tokenizer::~Tokenizer()
{}

void
Tokenizer::parse( const std::string& pString, std::tm& pDate, char pDateDelim, char pTimeDelim, char pDateTimeDelim )
{
    std::istringstream iss( pString );
    
    int year = 1900, month = 1, day = 0, hour = 0, min = 0, sec = 0;
    
    std::string dateString;
    std::string timeString;
    std::string s;
    
    // date
    if( getline(iss, dateString, pDateTimeDelim) )
    {
        std::istringstream diss( dateString );
        
        if( getline(diss, s, pDateDelim) ) parse(s, day); // day
        if( getline(diss, s, pDateDelim) ) parse(s, month); // month
        if( getline(diss, s, pDateDelim) ) parse(s, year); // year
    }
    
    // time
    if( getline(iss, timeString, pDateTimeDelim) )
    {
        std::istringstream tiss( timeString );
        
        if( getline(tiss, s, pTimeDelim) ) parse(s, hour); // hour
        if( getline(tiss, s, pTimeDelim) ) parse(s, min); // min
        if( getline(tiss, s, pTimeDelim) ) parse(s, sec); // sec
    }
    
    pDate.tm_year = year - 1900;
    pDate.tm_mon = month - 1;
    pDate.tm_mday = day;
    pDate.tm_hour = hour;
    pDate.tm_min = min;
    pDate.tm_sec = sec;
}

void
Tokenizer::parse( const std::tm& pDate, std::string& pString, char pDateDelim, char pTimeDelim, char pDateTimeDelim )
{
    std::stringstream ss;
    
	ss << std::setfill('0');
    ss << std::setw(2) <<  pDate.tm_mday << pDateDelim << std::setw(2) <<  pDate.tm_mon + 1 << pDateDelim << std::setw(4) << pDate.tm_year + 1900 << pDateTimeDelim << std::setw(2) << pDate.tm_hour << pTimeDelim << std::setw(2) << pDate.tm_min << pTimeDelim << std::setw(2) << pDate.tm_sec;
    
    pString = ss.str();
}

void
Tokenizer::substring( const std::string& pString, std::string& pResult, const std::string& pFrom, const std::string& pTo)
{
    size_t fromPos, toPos;
    fromPos = pString.find(pFrom);
    
    if( fromPos != std::string::npos )
    {
        if( pTo.size() == 0 || pString.find(pTo) == std::string::npos ) toPos = pString.size();
        else toPos = pString.find(pTo);
        pResult = pString.substr( fromPos + pFrom.size(), toPos - ( fromPos + pFrom.size() ) );
    }
    else
    {
        pResult = "";
    }
}

void
Tokenizer::split( const std::string& pString, std::vector< std::string >& pStrings, char pDelim )
{
    std::istringstream iss( pString );
    std::string s;
    
    pStrings.clear();
    
    while( getline(iss, s, pDelim) )
    {
        pStrings.push_back(s);
    }
}
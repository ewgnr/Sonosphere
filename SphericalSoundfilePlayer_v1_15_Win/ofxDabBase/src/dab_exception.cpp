/** \file dab_exception.cpp
*/

#include "dab_exception.h"
#include <sstream>

using namespace dab;

Exception::Exception( const std::string& pFile, const std::string& pFunction, int pLine )
{
    mFiles.push_back(pFile);
    mFunctions.push_back(pFunction);
    mLines.push_back(pLine);
}

Exception::Exception( const std::string& pMessage, const std::string& pFile, const std::string& pFunction, int pLine )
{
    mMessages.push_back(pMessage);
    mFiles.push_back(pFile);
    mFunctions.push_back(pFunction);
    mLines.push_back(pLine);
}

Exception::Exception( const Exception& pException )
: mMessages(pException.mMessages)
, mFiles(pException.mFiles)
, mFunctions(pException.mFunctions)
, mLines(pException.mLines)
{}

Exception::~Exception()
{}
    
Exception&
Exception::operator=( const Exception& pException )
{
    mMessages = pException.mMessages;
    mFiles = pException.mFiles;
    mFunctions = pException.mFunctions;
    mLines = pException.mLines;
    
    return *this;
}

Exception&
Exception::operator+= ( const Exception& pException )
{
    for(auto iter=pException.mMessages.begin(); iter != pException.mMessages.end(); ++iter) mMessages.push_back( (*iter) );
    for(auto iter=pException.mFiles.begin(); iter != pException.mFiles.end(); ++iter) mFiles.push_back( (*iter) );
    for(auto iter=pException.mFunctions.begin(); iter != pException.mFunctions.end(); ++iter) mFunctions.push_back( (*iter) );
    for(auto iter=pException.mLines.begin(); iter != pException.mLines.end(); ++iter) mLines.push_back( (*iter) );
    
    return *this;
}
    
Exception::operator std::string() const
{
    std::stringstream stream;
    
    int traceSize = mFiles.size();
    
    for(int tI=0; tI<traceSize; ++tI)
    {
        stream << "trace " << tI << " message " << mMessages[tI] << " function " << mFunctions[tI] << " file " << mFiles[tI] << " line " << mLines[tI] << "\n";
    }
    
    return stream.str();
}
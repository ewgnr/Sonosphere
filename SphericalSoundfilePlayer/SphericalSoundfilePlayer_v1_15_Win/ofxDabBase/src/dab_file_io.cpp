/** \file dab_file_io.cpp
*/

#include "dab_file_io.h"
#include <fstream>
#include <sstream>

using namespace dab;

FileIO::FileIO()
{}

FileIO::~FileIO()
{}

void
FileIO::write( const std::string& pString, const std::string& pFileName )
{
    std::ofstream fileStream( pFileName.c_str() ); // C++ 98
    
    if( fileStream.is_open() == false ) throw Exception( "File Error: failed to write to file " + pFileName, __FILE__, __FUNCTION__, __LINE__ );
    
    fileStream << pString;
    
    fileStream.close();
}

void
FileIO::read( const std::string& pFileName, std::string& pString )
{
    std::ifstream fileStream(pFileName.c_str(), std::ifstream::in); // C++ 98
    
    if( fileStream.is_open() == false ) throw Exception( "File Error: failed to read from file " + pFileName, __FILE__, __FUNCTION__, __LINE__ );
    
    std::stringstream ss;
    
    ss << fileStream.rdbuf();
    
    pString = ss.str();
}

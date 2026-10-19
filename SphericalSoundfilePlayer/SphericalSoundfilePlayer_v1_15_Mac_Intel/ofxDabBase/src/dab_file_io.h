/** \file dab_fileIO.h
*/

#pragma once

#include <iostream>
#include "dab_singleton.h"
#include "dab_exception.h"

namespace dab
{

class FileIO : public Singleton<FileIO>
{
public:
    friend class Singleton<FileIO>;
    
    void write( const std::string& pString, const std::string& pFileName );
    void read( const std::string& pFileName, std::string& pString );
    
protected:
    FileIO();
    ~FileIO();
};
    
};

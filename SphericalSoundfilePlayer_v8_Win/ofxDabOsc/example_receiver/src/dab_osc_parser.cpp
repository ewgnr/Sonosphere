/** \file dab_osc_parser.cpp
*/

#include "dab_osc_parser.h"

using namespace dab;

OscParser::OscParser()
{}

OscParser::~OscParser()
{}

void
OscParser::notify( std::shared_ptr<OscMessage> pMessage )
{
    std::cout << "OscParser::notify( std::shared_ptr<OscMessage> pMessage )\n";
    std::cout << "pMessage\n";
    std::cout << *pMessage << "\n";
    
    
//    
//    std::cout << "notify: message " << pMessage->address() << "\n";
//    
//    
//    const std::vector<_OscArg*>& args = pMessage->arguments();
//    int argCount = pMessage->argumentCount();
//    
//    std::cout << "address " << pMessage->address() << "\n";
//    std::cout << "argCount " << argCount << "\n";
//    
//    for(int aI=0; aI<argCount; ++aI)
//    {
//        _OscArg* arg = args[aI];
//        
//        std::cout << "arg " << aI << " : " << arg->info() << "\n";
//    }
}
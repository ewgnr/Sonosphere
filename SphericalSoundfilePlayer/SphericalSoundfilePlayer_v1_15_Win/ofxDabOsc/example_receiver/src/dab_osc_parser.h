/** \file dab_osc_parser.h
*/

#ifndef _dab_osc_parser_h_
#define _dab_osc_parser_h_

#include <iostream>
#include "dab_osc_receiver.h"

namespace dab
{
 
class OscParser : public OscListener
{
public:
    OscParser();
    ~OscParser();
    
    void notify( std::shared_ptr<OscMessage> pMessage );
    
protected:
};
    
};

#endif

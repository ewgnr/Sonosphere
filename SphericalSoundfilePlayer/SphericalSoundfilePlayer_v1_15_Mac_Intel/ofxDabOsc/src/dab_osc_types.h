/** \file dab_osc_types.h
*/

#ifndef _dab_osc_types_h_
#define _dab_osc_types_h_

namespace dab
{
    
//#define OSC_BUFFER_SIZE 1024
#define OSC_BUFFER_SIZE 4096
    
    
#pragma mark oscArgType definiton
    
    enum OscType
    {
        OSC_TYPE_TRUE,
        OSC_TYPE_FALSE,
        OSC_TYPE_CHAR,
        OSC_TYPE_INT32,
        OSC_TYPE_INT64,
        OSC_TYPE_FLOAT,
        OSC_TYPE_DOUBLE,
        OSC_TYPE_STRING,
        OSC_TYPE_SYMBOL,
        OSC_TYPE_MIDI_MESSAGE,
        OSC_TYPE_TRIGGER,
        OSC_TYPE_TIMETAG,
        OSC_TYPE_BLOB,
        OSC_TYPE_BUNDLE,
        OSC_TYPE_RGBA_COLOR,
        OSC_TYPE_ARRAY_BEGIN,
        OSC_TYPE_ARRAY_END,
        EXT_TYPE_ARG_CHAR_ARRAY,
        EXT_TYPE_ARG_INT32_ARRAY,
        EXT_TYPE_ARG_INT64_ARRAY,
        EXT_TYPE_ARG_FLOAT_ARRAY,
        EXT_TYPE_ARG_DOUBLE_ARRAY,
        OSC_TYPE_NONE
    };
    
#pragma mark oscTypeTag definiton
    
    enum OscTag
    {
        OSC_TAG_TRUE = 'T',
        OSC_TAG_FALSE = 'F',
        OSC_TAG_CHAR = 'c',
        OSC_TAG_INT32 = 'i',
        OSC_TAG_INT64 = 'h',
        OSC_TAG_FLOAT = 'f',
        OSC_TAG_DOUBLE = 'd',
        OSC_TAG_STRING = 's',
        OSC_TAG_SYMBOL = 'S',
        OSC_TAG_MIDI_MESSAGE = 'm',
        OSC_TAG_TRIGGER = 'I',
        OSC_TAG_TIMETAG = 't',
        OSC_TAG_BLOB = 'b',
        OSC_TAG_BUNDLE = 'B',
        OSC_TAG_RGBA_COLOR = 'r',
        OSC_TAG_ARRAY_BEGIN= '[',
        OSC_TAG_ARRAY_END = ']',
        EXT_TAG_ARG_CHAR_ARRAY = 'c',
        EXT_TAG_ARG_INT32_ARRAY = 'i',
        EXT_TAG_ARG_INT64_ARRAY = 'h',
        EXT_TAG_ARG_FLOAT_ARRAY = 'f',
        EXT_TAG_ARG_DOUBLE_ARRAY = 'd',
        OSC_TAG_NONE = 'N'
    };
};

#endif

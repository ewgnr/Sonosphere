/*
    oscpack -- Open Sound Control (OSC) packet manipulation library
    http://www.rossbencina.com/code/oscpack

    Copyright (c) 2004-2013 Ross Bencina <rossb@audiomulch.com>

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files
    (the "Software"), to deal in the Software without restriction,
    including without limitation the rights to use, copy, modify, merge,
    publish, distribute, sublicense, and/or sell copies of the Software,
    and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
    ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
    WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/*
    The text above constitutes the entire oscpack license; however,
    the oscpack developer(s) also make the following non-binding requests:

    Any person wishing to distribute modifications to the Software is
    requested to send the modifications to the original developer so that
    they can be incorporated into the canonical version. It is also
    requested that these non-binding requests be included whenever the
    above license is reproduced.
*/
#ifndef INCLUDED_OSCPACK_OSCOUTBOUNDPACKETSTREAM_H
#define INCLUDED_OSCPACK_OSCOUTBOUNDPACKETSTREAM_H

#include <cstring> // for size_t
#include <string> // <-- added for std::string

#include "OscException.h"
#include "OscTypes.h"

namespace osc {

class OutOfBufferMemoryException : public Exception {
public:
	OutOfBufferMemoryException(const char * w = "out of buffer memory")
		: Exception(w) { }
};

class BundleNotInProgressException : public Exception {
public:
	BundleNotInProgressException(
		const char * w = "call to EndBundle when bundle is not in progress")
		: Exception(w) { }
};

class MessageInProgressException : public Exception {
public:
	MessageInProgressException(
		const char * w = "opening or closing bundle or message while message is in progress")
		: Exception(w) { }
};

class MessageNotInProgressException : public Exception {
public:
	MessageNotInProgressException(
		const char * w = "call to EndMessage when message is not in progress")
		: Exception(w) { }
};

class OutboundPacketStream {
public:
	OutboundPacketStream(char * buffer, std::size_t capacity);
	~OutboundPacketStream();

	void Clear();

	std::size_t Capacity() const;
	std::size_t Size() const;
	const char * Data() const;

	bool IsReady() const;
	bool IsMessageInProgress() const;
	bool IsBundleInProgress() const;

	OutboundPacketStream & operator<<(const BundleInitiator & rhs);
	OutboundPacketStream & operator<<(const BundleTerminator & rhs);
	OutboundPacketStream & operator<<(const BeginMessage & rhs);
	OutboundPacketStream & operator<<(const MessageTerminator & rhs);

	OutboundPacketStream & operator<<(bool rhs);
	OutboundPacketStream & operator<<(const NilType & rhs);
	OutboundPacketStream & operator<<(const InfinitumType & rhs);
	OutboundPacketStream & operator<<(int32 rhs);

#if !(defined(__x86_64__) || defined(_M_X64) || defined(__LP64__))
	OutboundPacketStream & operator<<(int rhs) {
		*this << (int32)rhs;
		return *this;
	}
#endif

	OutboundPacketStream & operator<<(float rhs);
	OutboundPacketStream & operator<<(char rhs);
	OutboundPacketStream & operator<<(const RgbaColor & rhs);
	OutboundPacketStream & operator<<(const MidiMessage & rhs);
	OutboundPacketStream & operator<<(int64 rhs);
	OutboundPacketStream & operator<<(const TimeTag & rhs);
	OutboundPacketStream & operator<<(double rhs);
	OutboundPacketStream & operator<<(const char * rhs);
	OutboundPacketStream & operator<<(const Symbol & rhs);
	OutboundPacketStream & operator<<(const Blob & rhs);
	OutboundPacketStream & operator<<(const ArrayInitiator & rhs);
	OutboundPacketStream & operator<<(const ArrayTerminator & rhs);

	// -----------------------------------------------------------------
	// DAB Extension: Fix ambiguous operator<< overloads for extra types
	// -----------------------------------------------------------------

	OutboundPacketStream & operator<<(const std::string & rhs) {
		*this << rhs.c_str();
		return *this;
	}

	OutboundPacketStream & operator<<(unsigned int rhs) {
		*this << static_cast<int32>(rhs);
		return *this;
	}

	OutboundPacketStream & operator<<(long rhs) {
		*this << static_cast<int64>(rhs);
		return *this;
	}

	OutboundPacketStream & operator<<(unsigned long rhs) {
		*this << static_cast<int64>(rhs);
		return *this;
	}

	template <typename T>
	OutboundPacketStream & operator<<(const T &) {
		static_assert(!sizeof(T),
			"Unsupported type passed to osc::OutboundPacketStream <<");
		return *this;
	}

private:
	char * BeginElement(char * beginPtr);
	void EndElement(char * endPtr);
	bool ElementSizeSlotRequired() const;
	void CheckForAvailableBundleSpace();
	void CheckForAvailableMessageSpace(const char * addressPattern);
	void CheckForAvailableArgumentSpace(std::size_t argumentLength);

	char * data_;
	char * end_;
	char * typeTagsCurrent_;
	char * messageCursor_;
	char * argumentCurrent_;
	uint32 * elementSizePtr_;
	bool messageIsInProgress_;
};

} // namespace osc

#endif /* INCLUDED_OSCPACK_OSCOUTBOUNDPACKETSTREAM_H */

## ofxDabBase

Daniel Bisig - Coventry University, UK - [ad5041@coventry.ac.uk](ad5041@coventry.ac.uk) - Zurich University of the Arts, CH - [daniel.bisig@zhdk.ch](daniel.bisig@zhdk.ch)

---

## Summary

ofxDabBase provides some basic functionality in the form of classes that deal with multidimensional data, text file parsing, and information flow. The code is compatible with OpenFrameworks 0.11 and has been tested on Windows and MacOS. The following classes are available.

### Data Structure

**Array**: a one-dimensional array for scalar values of arbitrary type, supports dynamic size changes. 

**Buffer**: a one-dimensional array for vector values of arbitrary type, supports dynamic size changes. 

**IndexMap**; a data structure that combines properties of a dictionary and vector. 

**RingBuffer**: a ring buffer for values of arbitrary type. 

**Value**: a wrapper for scalar values or arrays of arbitrary type



### Text Files

**FileIO**: provides functionality for reading and writing text files.

**Tokenizer**: provides functionality for converting text in scalar or vector values.



### Information Flow

**Exception**: handles exception information 

**UpdateListener** and **UpdateNotifier**: base classes for handling update notifications.

**Parameter**: a data wrapper that provides value ranges and handles value change notifications



### Others

**Singleton**: a base class that ensures that only one instance of a class exists
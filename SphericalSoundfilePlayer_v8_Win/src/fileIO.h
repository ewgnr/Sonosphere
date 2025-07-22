#pragma once

#include <iostream>
#include <fstream>
#include <sstream>

class fileIO
{
public:
	fileIO() {}
	~fileIO() {}

	void read(const std::string& pFileName, std::string& pString)
	{
		std::ifstream fileStream(pFileName.c_str());
		std::stringstream ss;
		ss << fileStream.rdbuf();
		pString = ss.str();
	}

	void write(const std::string& pString, const std::string& pFileName)
	{
		std::ofstream fileStream(pFileName.c_str());
		fileStream << pString;
		fileStream.close();
	}
};
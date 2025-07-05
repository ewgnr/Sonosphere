#pragma once

#include <iostream>
#include <fstream>
#include <sstream>

class fileIO
{
public:
	fileIO() {}
	~fileIO() {}

	bool read(const std::string& pFileName, std::string& pString)
	{
		try
		{
			std::ifstream fileStream(pFileName.c_str());
			fileStream.exceptions(std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit);

			while (fileStream)
			{
				std::stringstream ss;
				ss << fileStream.rdbuf();
				pString = ss.str();

				return true;
			}
			fileStream.close();
		}
		catch (const std::exception& e) 
		{
			std::cout << "Can't load file with exeption : " << e.what() << std::endl;

			return false;
		}
	}

	void write(const std::string& pString, const std::string& pFileName)
	{
		std::ofstream fileStream(pFileName.c_str());
		fileStream << pString;
		fileStream.close();
	}
};
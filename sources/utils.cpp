#include "utils.h"
#include <cassert>
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
using std::vector;
using std::string;

#ifdef __linux__
std::string getFullPath(const char * partialPath)
{
	char full[FILENAME_MAX];
	if (realpath (partialPath, full ) != NULL){
		return std::string(full);
	}
	else{
		printf("Invalid path\n");
		throw std::invalid_argument("Incorrect path");

	}
}
#elif __WIN32|| __WIN64
std::string getFullPath(const char * partialPath)
{
	char full[_MAX_PATH];
	if (_fullpath(full, partialPath, _MAX_PATH) != NULL){
		return std::string(full);
	}
	else{
		printf("Invalid path\n");
		throw std::exception("Incorrect path");

	}
}
#endif

const char *oclErrorCode(cl_int code)
{
	std::map<cl_int, std::string>::const_iterator iter = oclErrorCodes.find(code);
	if (iter == oclErrorCodes.end())
		return "UNKNOWN ERROR";
	else
		return iter->second.c_str();
}
std::string get_kernel_as_string(const std::string& filename)
{
	std::ifstream kernel_file(filename);
	//assert(kernel_file.good());
	std::string kernel_source(
		std::istreambuf_iterator<char>(kernel_file),
		(std::istreambuf_iterator<char>()));
	assert(kernel_source.length() > 0);
	return kernel_source;
}
std::uniform_int_distribution<int> getRandIntGenerator(int min, int max)
{
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_int_distribution<> dis(min, max);
	return dis;
}
std::uniform_real_distribution<float> getRandFloatGenerator(float min, float max)
{
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<float> dis(1, 6);
	return dis;

}
std::uniform_real_distribution<double> getRandDoubleGenerator(double min, double max)
{
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<double> dis(min,max);
	return dis;

}
std::uniform_real_distribution<long double> getRandLongDoubleGenerator(long double min, long double max)
{
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<long double> dis(min, max);
	return dis;

}

std::string stringToLower(std::string str)
{
	using std::string;	
	assert(str.length() > 0);
	string s;
	s.resize(str.size());
	std::transform(str.begin(), str.end(), s.begin(), ::tolower);
	assert(s.length() > 0);
	return s;
}

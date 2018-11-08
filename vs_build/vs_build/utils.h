#pragma once
#include <random>
#include <string>
#include <vector>
using std::vector;
std::uniform_int_distribution<int> getRandIntGenerator(int min, int max);
std::uniform_real_distribution<float> getRandFloatGenerator(float min, float max);
std::uniform_real_distribution<double> getRandDoubleGenerator(double min, double max);
std::uniform_real_distribution<long double> getRandLongDoubleGenerator(long double min, long double max);
std::string stringToLower(std::string str);
std::string get_kernel_as_string(const std::string& path);
std::string getFullPath(const char * partialPath);
template <typename T>
vector<vector<T>> createFifos(int count, int size, T defaultValue){
	vector<vector<T>> fifos(count);
	for (auto &x : fifos){
		x = vector<T>(size, defaultValue);
	}
	return fifos;
}

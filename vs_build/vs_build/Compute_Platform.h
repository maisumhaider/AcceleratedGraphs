#pragma once
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>
#include <vector>
#include <string>
#include <tuple>
using std::vector;
using std::string;
using std::tuple;
class Compute_Platform
{
	vector<tuple<string, string>> target_devices;
	vector<cl::Platform> all_platforms;
	vector<cl::Device> all_devices;
	vector<cl::Context> contexts;
public:
	Compute_Platform();
	explicit Compute_Platform(string& synthesis_file);
	const vector<cl::Device>& get_devices() const { return all_devices; }
	const vector<cl::Context>& get_context() const { return contexts; }

	const cl::Platform& get_platforms(const int id) { return all_platforms[id]; }
	const vector<cl::Platform>& getAllPlatforms() const { return all_platforms; }
	const cl::Context& get_context(const string vendor, const cl_int device_type);
	const cl::CommandQueue& get_device_queue(const string vendor, const cl_int device_type);
	void get_devices(const cl_int device_type,vector<cl::Device>& devices);
	void get_devices(const string& vendor, vector<cl::Device>& devices);
	void get_devices(const string& vendor, const cl_int device_type, vector<cl::Device>& devices); //Assumes One type of device per vendor
	~Compute_Platform();
};
#include "Compute_Platform.h"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <locale>
#include "utils.h"
#include <iso646.h>
#include "../../pugixml-1.9/pugixml-1.9/src/pugixml.hpp"
using std::vector;
using std::string;
using std::tuple;
using std::cout;
using std::endl;
void get_target_platforms_from_xml(const string& synthesis_file, vector<tuple<string, string>>& target_platform)
{
	pugi::xml_document doc;
	pugi::xml_parse_result result = doc.load_file(synthesis_file.c_str());
	if (result.status != pugi::xml_parse_status::status_ok){
		std::cout << "Failed to open " << synthesis_file << std::endl;
		exit(EXIT_FAILURE);
	}
	pugi::xml_node devices = doc.child("platform").child("devices");
	for (pugi::xml_node device = devices.first_child(); device; device = devices.next_sibling())
	{
			target_platform.emplace_back(device.attribute("vendor").value(), device.attribute("type").value());
	}


}
unsigned long long get_device_type_from_string(string device_type)
{
	if (device_type == "cpu") return CL_DEVICE_TYPE_CPU;
	else if (device_type == "gpu") return CL_DEVICE_TYPE_CPU;
	else if (device_type == "accel") return CL_DEVICE_TYPE_ACCELERATOR;
	else return CL_DEVICE_TYPE;

}
bool device_is_target(const tuple<string, string>& target, const cl::Device& device)
{
	string target_vendor = stringToLower(std::get<0>(target));
	string device_vendor = stringToLower(device.getInfo<CL_DEVICE_NAME>());
	auto target_device_type = get_device_type_from_string(stringToLower(std::get<1>(target)));
	auto device_type = device.getInfo<CL_DEVICE_TYPE>();
	return device_vendor.find(target_vendor) != string::npos && device_type == target_device_type;

}
bool checkDeviceVendor(string vendor, cl::Device device)
{
	return stringToLower(device.getInfo<CL_DEVICE_VENDOR>()).find(vendor) != string::npos;
}
void createCommandQueueForContext(cl::Context ctx)
{
	cl_int err;
	auto device = ctx.getInfo<CL_CONTEXT_DEVICES>();
	assert(device.size() == 1);
	if (checkDeviceVendor("nvidia",device.front())) return;
	cl::CommandQueue queue(ctx, 0, &err);
	assert(err==CL_SUCCESS);
	
}

Compute_Platform::Compute_Platform()
{
	std::cout << "Initializing platform." << std::endl;
	vector<cl::Device> temp;
	cl::Platform::get(&all_platforms);
	all_platforms.erase(std::remove_if(all_platforms.begin(), all_platforms.end(),
		[](cl::Platform platform){return stringToLower(platform.getInfo<CL_PLATFORM_NAME>()).find("experimental") != string::npos; }), all_platforms.end());
	for (auto &platform : all_platforms)
	{	
		cl_int err = platform.getDevices(CL_DEVICE_TYPE_ALL, &temp);
		assert(err==CL_SUCCESS);
		all_devices.insert(all_devices.end(), temp.begin(), temp.end());		
	}
	for (const cl::Device &device : all_devices)
	{
		if (checkDeviceVendor("nvidia", device)) continue;
		cl_int err;
		cl::Context ctx(device, nullptr, nullptr, nullptr, &err);
		assert(err == CL_SUCCESS);
		contexts.push_back(ctx);
		
	}
}



Compute_Platform::Compute_Platform(string& synthesis_file)
{
	get_target_platforms_from_xml(synthesis_file, target_devices);
	std::cout << "Initializing platform." << std::endl;
	vector<cl::Device> temp;
	cl::Platform::get(&all_platforms);
	all_platforms.erase(std::remove_if(all_platforms.begin(), all_platforms.end(),
		[](cl::Platform platform)
	{
		return stringToLower(platform.getInfo<CL_PLATFORM_NAME>()).find("experimental") != string::npos;
	}), all_platforms.end());
	all_platforms.shrink_to_fit();
	for (auto &platform : all_platforms)
	{
		cl_int err = platform.getDevices(CL_DEVICE_TYPE_ALL, &temp);
		assert(err == CL_SUCCESS);
	}
	for (auto device : temp)
	{
		for (auto target : target_devices)
		{
			if (device_is_target(target, device))
			{
								
				all_devices.push_back(device);
			}
		}
	}
	for (const cl::Device &device : all_devices)
	{
		//TODO Remove nvidia ignore.
		if (checkDeviceVendor("nvidia", device)) continue;
		cl_int err;
		cout << "Creating Context for Device " << device.getInfo<CL_DEVICE_NAME>(&err) << endl;
		assert(err == CL_SUCCESS);
		cl::Context ctx(device, nullptr, nullptr, nullptr, &err);
		assert(err == CL_SUCCESS);
		contexts.push_back(ctx);
	}
}

const cl::Context& Compute_Platform::get_context(string vendor, cl_int device_type) 
{
	for (int i = 0; i < contexts.size();i++)
	{
		vector<cl::Device> devices = contexts[i].getInfo<CL_CONTEXT_DEVICES>();
		for (const auto& device : devices)
		{
			if (checkDeviceVendor(vendor, device) && device.getInfo<CL_DEVICE_TYPE>() == device_type) return contexts[i];
		}
	}
	return {};
}



void  Compute_Platform::get_devices(cl_int device_type, vector<cl::Device>& devices)
{
	for (auto &device : all_devices)
	{
		if (device.getInfo<CL_DEVICE_TYPE>() == device_type){
			devices.push_back(device);
		}
	} 
}
void Compute_Platform::get_devices(const string& vendor, vector<cl::Device>& devices)
{
	std::locale loc;
	for (auto &device : all_devices)
	{
		if (checkDeviceVendor(vendor, device)){
			devices.push_back(device);
		}
	}
}
void Compute_Platform::get_devices(const string& device_vendor_name, cl_int device_type, vector<cl::Device>& devices)
{
	get_devices(device_vendor_name,devices);
	assert(not devices.empty() && " Device vendor not found");
	vector<cl::Device> desired_devices;
	for (auto &device: devices)
	{
		if (device.getInfo<CL_DEVICE_TYPE>()==device_type)
		{
			desired_devices.push_back(device);
		}
	}
}
Compute_Platform::~Compute_Platform()
{
	contexts.clear();
	all_devices.clear();
	all_platforms.clear();
}

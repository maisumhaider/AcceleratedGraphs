#pragma once
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>
#include <vector>
#include <string>
using std::vector;
class ComputationNode
{
	cl::Kernel node;
	//TODO  rewrite for heterogenous node.
	vector<vector<int>> kernel_data_in_vector_buffers;
	vector<vector<int>> kernel_data_out_vector_buffers;
	vector<cl::Buffer> kernel_input_buffers_;
	vector<cl::Buffer> kernel_output_buffers_;
public:
	ComputationNode();
	ComputationNode(cl::Program& program, std::string kernel_name);
	void set_CL_Buffers();
	vector<vector<int>>& get_kernel_input_data_buffers()  { return kernel_data_in_vector_buffers; }
	vector<vector<int>>& get_kernel_output_data_buffers() { return kernel_data_out_vector_buffers; }
	const vector<cl::Buffer>& get_kernel_input_buffers() const { return kernel_input_buffers_; }
	const vector<cl::Buffer>& get_kernel_output_buffers() const { return kernel_output_buffers_; }
	const cl::Kernel& get_kernel() const { return node; }
	~ComputationNode();
	template <class T>
	void add_input_data_buffer(vector<T> data_buffer)
	{
		kernel_data_in_vector_buffers.push_back(data_buffer);
	}

	template <class T>
	void add_output_data_buffer(vector<T> data_buffer)
	{
		kernel_data_out_vector_buffers.push_back(data_buffer);
	}

};



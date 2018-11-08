#include "ComputationNode.hpp"
#include "ComputationGraph.h"
#include <string>
#include <cassert>
using std::string;
ComputationNode::ComputationNode()
= default;


ComputationNode::ComputationNode(cl::Program& program, string kernel_name) 
{
	cl_int err;
	int i = 0;
	node = cl::Kernel(program, kernel_name.c_str(), &err);
	cl::Context ctx = program.getInfo<CL_PROGRAM_CONTEXT>();
	
}
void ComputationNode::set_CL_Buffers()
{
	cl::Context ctx = node.getInfo<CL_KERNEL_CONTEXT>();
	cl_int err;
	int i =0;
	for (auto & data_in_buffer : kernel_data_in_vector_buffers)
	{
		auto size = sizeof(int)*data_in_buffer.size();
		//TODO remove hard coded data type
		cl::Buffer inp(ctx, CL_MEM_READ_ONLY, size,nullptr, &err);
		assert(err == CL_SUCCESS);
		kernel_input_buffers_.emplace_back(inp);
		node.setArg(i++, inp);
	}
	for (auto & data_out_buffer : kernel_data_out_vector_buffers)
	{
		//TODO remove hard coded data type
		auto size = sizeof(int)*data_out_buffer.size();
		cl::Buffer op(ctx, CL_MEM_READ_ONLY, size, nullptr, &err);
		assert(err == CL_SUCCESS);
		kernel_output_buffers_.emplace_back(op);
		node.setArg(i++, op);
	}
}
ComputationNode::~ComputationNode()
{
	kernel_input_buffers_.clear();
	kernel_output_buffers_.clear();
}


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
	for (auto & input_buffer : kernel_input_buffers_)
	{
		err = node.setArg(i++, *input_buffer);
		assert(err == CL_SUCCESS);
	}
	for (auto & output_buffer : kernel_output_buffers_)
	{
		//TODO remove hard coded data type
		err = node.setArg(i++, *output_buffer);
		assert(err == CL_SUCCESS);
	}
}
ComputationNode::~ComputationNode()
{
	kernel_input_buffers_.clear();
	kernel_output_buffers_.clear();
}


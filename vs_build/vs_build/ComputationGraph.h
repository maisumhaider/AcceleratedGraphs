#pragma once
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>
#include <vector>
#include <string>
#include "ComputationNode.hpp"
#include "Compute_Platform.h"
#include "CompGraphParser.h"
using std::vector;
class ComputationGraph
{  
	cl::Program program;
	CompGraphParser parser;
	map<string, vector<int>> fifos;
	map<string, cl::Buffer> buffers;
	vector<ComputationNode> computation_nodes;
	
public:
	ComputationGraph();
	ComputationGraph(const string& comp_graph, Compute_Platform& platform);
	const cl::Program& get_program()const { return  program; }
	const vector<ComputationNode>& get_nodes() const { return computation_nodes; }
	int get_fifo_count() const { return fifos.size(); }
	int get_buffer_count () const { return buffers.size(); }
	ComputationGraph(std::string &path, bool Debug, Compute_Platform& platform);
	ComputationGraph(std::string &path, bool Debug, cl::Context &context);
	~ComputationGraph();
};


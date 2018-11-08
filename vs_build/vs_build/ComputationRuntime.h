#pragma once
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>
#include <vector>
#include "Compute_Platform.h"
#include "ComputationGraph.h"
using std::vector;
class ComputationRuntime
{
	vector<cl::CommandQueue> queues;
public:
	ComputationRuntime() = default;
	ComputationRuntime(Compute_Platform& platform);
	const cl::CommandQueue& get_command_queue() const;
	bool execute_compute_graph(ComputationGraph& graph);
	~ComputationRuntime();
};


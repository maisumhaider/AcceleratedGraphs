#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL\cl2.hpp>
#include <iostream>
#include <string>
#include "vs_build/vs_build/utils.h"
#include "vs_build/vs_build/Compute_Platform.h"
#include "vs_build/vs_build/ComputationGraph.h"
#include "vs_build/vs_build/ComputationRuntime.h"
#include "pugixml-1.9/pugixml-1.9/src/pugixml.hpp"
#include <cassert>
#define __RELATIVE_KERNEL_FOLDER "..\\..\\Kernels\\"
#define __TASK_FOLDER "..\\..\\Task\\"
using std::cout;
using std::string;
using std::endl;
int main(int argc, char* argv[])
{
	assert(argc == 4);
	cout << "Number of input files " << argc << endl;
	auto comp_graph_file = getFullPath((string(__TASK_FOLDER)+argv[1]).c_str());
	auto synthesis_file = getFullPath((string(__TASK_FOLDER) + argv[2]).c_str());
	auto schedule_file = getFullPath((string(__TASK_FOLDER) + argv[3]).c_str());

	

	string kernel_path = getFullPath((string(__RELATIVE_KERNEL_FOLDER) + "add.cl").c_str());
	Compute_Platform test = Compute_Platform(synthesis_file);	
	ComputationGraph graph = ComputationGraph(comp_graph_file,test);
	ComputationRuntime runtime = ComputationRuntime(test);
	bool match = runtime.execute_compute_graph(graph);
	cout << "Results " << (match ? "matched" : "mismatched") << endl;
	system("pause");
	exit(match ? EXIT_SUCCESS : EXIT_FAILURE);
}
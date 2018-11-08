#include "ComputationGraph.h"
#include <string>
#include <cassert>
#include "utils.h"
#include "../../pugixml-1.9/pugixml-1.9/src/pugixml.hpp"
#include <iostream>
#define __RELATIVE_KERNEL_FOLDER "..\\..\\Kernels\\"
using std::string;

ComputationGraph::ComputationGraph()
= default;

ComputationGraph::ComputationGraph(const string& comp_graph, Compute_Platform& platform)
{
	
	string kernel_folder = __RELATIVE_KERNEL_FOLDER;
	string kernel_src = "";
	pugi::xml_document doc;
	pugi::xml_parse_result result = doc.load_file(comp_graph.c_str());
	if (result.status != pugi::xml_parse_status::status_ok){
		std::cout << "Failed to open " << comp_graph << std::endl;
		exit(EXIT_FAILURE);
	}
	parser = CompGraphParser(doc);
	pugi::xml_node nodes = doc.child("graph").child("nodes");
	auto operations = parser.get_worker_nodes();
	assert(operations.empty()==false);
	for (auto& operation: operations)
	{

		kernel_src += get_kernel_as_string(getFullPath((kernel_folder + operation.second.subtype + ".cl").c_str())) + "\n";
	}
	cl_int err;
	program = cl::Program(platform.get_context().front(), kernel_src, true, &err);
	assert(err == CL_SUCCESS);
	for (auto& operation : operations)
	{
		//TODO Change from hardcoded adding of input and output buffers to xml based
		auto computation_node = ComputationNode(program, operation.second.subtype);
		parser.set_comp_node_data_buffers(computation_node, operation.first);
		computation_node.set_CL_Buffers();
		computation_nodes.emplace_back(computation_node);
	}
	doc.reset();
}
ComputationGraph::ComputationGraph(std::string& path, bool Debug, Compute_Platform& platform)
{
	cl_int err;
	string kernel_src = get_kernel_as_string(path);
	program = cl::Program(platform.get_context().front(), kernel_src, false, &err);
	err = program.build((" -g -s " + path).c_str());
	assert(err == CL_SUCCESS);
	computation_nodes.emplace_back(ComputationNode(program, "add"));
}

ComputationGraph::ComputationGraph(std::string& path, bool Debug, cl::Context& context)
{
	cl_int err;
	string kernel_src = get_kernel_as_string(path);
	program = cl::Program(context, kernel_src, false, &err);
	err = program.build((" -g -s "+path).c_str());
	assert(err == CL_SUCCESS);
	computation_nodes.emplace_back(ComputationNode(program, "add"));
}

ComputationGraph::~ComputationGraph()
{
}



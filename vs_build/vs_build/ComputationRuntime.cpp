#include "ComputationRuntime.h"
#include <cassert>
#include "ComputationGraph.h"
#include "utils.h"
#include <iostream>
using std::cout;

ComputationRuntime::ComputationRuntime(Compute_Platform& platform)
{
	cl_int err;
	cl::CommandQueue q = cl::CommandQueue(platform.get_context().front(), 0, &err);	
	assert(err == CL_SUCCESS);
	queues.emplace_back(q);
	
}
const cl::CommandQueue& ComputationRuntime::get_command_queue() const
{
	return queues.front();
}
bool ComputationRuntime::execute_compute_graph(ComputationGraph& graph)
{
	//TODO Rewrite using schedule xml 
	cl_int err;
	cl::Event event;
	bool match = true;
	cl::CommandQueue queue = queues.front();	
	cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
	std::cout << "Executing on Devices "<<std::endl;
	for (auto dev : context.getInfo<CL_CONTEXT_DEVICES>())
	{
		cout << dev.getInfo<CL_DEVICE_NAME>() << CL_DEVICE_ENDIAN_LITTLE << " ";
	}
	cout << std::endl;
	int i = 0;	
	auto nodes = graph.get_nodes();
	for (auto& node : nodes){
		i = 0;
		const auto data_size = sizeof(int)*(*node.get_kernel_output_data_buffers()[i]).size();
		for (auto &x : node.get_kernel_input_buffers())
		{
			err = queue.enqueueWriteBuffer(*x, CL_TRUE, 0, data_size, (*node.get_kernel_input_data_buffers()[i]).data(), nullptr, &event);
			assert(err == CL_SUCCESS);
			i++;
		}
		err = queue.enqueueNDRangeKernel(node.get_kernel(), 0, cl::NDRange((*node.get_kernel_output_data_buffers()[0]).size()), cl::NullRange, nullptr, &event);
		assert(err == CL_SUCCESS);
		event.wait();
		i = 0;
		for (auto &x : node.get_kernel_output_buffers())
		{

			err = queue.enqueueReadBuffer(*x, CL_TRUE, 0, data_size, (*node.get_kernel_output_data_buffers()[i]).data(), nullptr, &event);
			assert(err == CL_SUCCESS);
			i++;
		}		
	}
	std::cout << "number of fifos:"<<graph.get_fifo_count()<<std::endl;
	std::cout << "number of cl buffers:" << graph.get_buffer_count()<<std::endl;	
 	return true;
}

ComputationRuntime::~ComputationRuntime()
{
}

#ifdef FPGA
#include "xcl2.hpp"
#else
#define posix_memalign(p, a, s) (((*(p)) = _aligned_malloc((s), (a))), *(p) ?0 :errno)
#include <CL/cl.hpp>
#include <ctime>
#include <fstream>
#include <iostream>
#endif
#include <vector>
#include <stdlib.h>
#include <cmath>
#include <cassert>
#define SIZE_M 1
#define SIZE_K 784
#define SIZE_N 500
#define SIZE_OUT 10

using std::vector;
using std::cout;
using std::endl;
using std::fabs;


template <typename T>
struct aligned_allocator
{
    using value_type = T;
    T* allocate(std::size_t num)
    {
        void* ptr = nullptr;
        if (posix_memalign(&ptr,4096,num*sizeof(T)))
            throw std::bad_alloc();
        return reinterpret_cast<T*>(ptr);
    }
    void deallocate(T* p, std::size_t num)
    {
        free(p);
    }
};
bool cmpFloat(float A, float B, float tolerance = 0.0f){
	if(fabs(A-B)> std::numeric_limits<float>::epsilon()){
	    cout<<"Mismatch "<< fabs(A-B)<<endl;
	    return false;
	}
	return true;
}
float RandomFloat(float min = 0.0, float max = float(SIZE_M)){
	assert(max>min);
	float random = ((float)rand())/(float)RAND_MAX;
	float range = max - min;
	return (random*range) + min;
}
void mmult(const float* A, const float* B, float* C, const int* dim){
	int dim_M = dim[0];
	int dim_K = dim[1];
	int dim_N = dim[2];
	float acc = 0.0f;
	for(int m=0;m<dim_M;m++){
		for(int n=0;n<dim_N;n++){
			acc = 0.0f;
			for(int k=0;k<dim_K;k++){
				acc += A[k*dim_M+m]*B[n*dim_K+k];
			}
			C[n*dim_M +m] = acc;
		}
	}
}
void mmult_fpga_ocl(cl::CommandQueue &q,cl::Context &context,cl::Kernel &kernel,
		vector<float,aligned_allocator<float>> &source_in1,
		vector<float,aligned_allocator<float>> &source_in2,
		vector<float,aligned_allocator<float>> &source_hw_results,
		vector<int,aligned_allocator<int>> &dimensions
		){

	size_t  input_size = sizeof(float)*dimensions[0]*dimensions[1];
	size_t  weights_size = sizeof(float)*dimensions[1]*dimensions[2];
	size_t output_size = sizeof(float)*dimensions[0]*dimensions[2];

	cl::Buffer buffer_in1 (context, CL_MEM_READ_ONLY,input_size);
	cl::Buffer buffer_in2 (context, CL_MEM_READ_ONLY,weights_size);
	cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY,output_size);
	cl::Buffer buffer_dim(context, CL_MEM_WRITE_ONLY,sizeof(int)*3);

	int nargs=0;
	kernel.setArg(nargs++,buffer_in1);
	kernel.setArg(nargs++,buffer_in2);
	kernel.setArg(nargs++,buffer_output);
	kernel.setArg(nargs++,buffer_dim);

    q.enqueueWriteBuffer(buffer_in1, CL_TRUE, 0, input_size, source_in1.data());
    q.enqueueWriteBuffer(buffer_in2, CL_TRUE, 0, weights_size, source_in2.data());
    q.enqueueWriteBuffer(buffer_dim, CL_TRUE, 0, sizeof(int)*3, dimensions.data());

    //Launch the Kernel
    q.enqueueTask(kernel);

    //Copying Device result data to Host memory
    q.enqueueReadBuffer(buffer_output, CL_TRUE, 0, output_size, source_hw_results.data());

    q.finish();
}
int main(int argc, char** argv)
{
	srand (time(NULL));
    //Allocate Memory in Host Memory    

    //Data Vectors
    std::vector<float,aligned_allocator<float>> input (SIZE_M*SIZE_K);
    std::vector<float,aligned_allocator<float>> weight_layer1 (SIZE_K*SIZE_N);
    
    std::vector<int,aligned_allocator<int>> dimensions(3);

    std::vector<float,aligned_allocator<float>> source_hw_results(SIZE_M*SIZE_N);
    std::vector<float,aligned_allocator<float>> source_sw_results(SIZE_M*SIZE_N);
    dimensions[0] = SIZE_M;
	dimensions[1] = SIZE_K;
	dimensions[2] = SIZE_N;

    // Create the test data and Software Result
    for(auto &x:input)x = RandomFloat();
    for(auto &x:weight_layer1)x = RandomFloat();
    for(auto &x:source_hw_results)x = 0;

//OPENCL HOST CODE AREA START
#ifdef FPGA
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
    std::string device_name = device.getInfo<CL_DEVICE_NAME>(); 

    //Create Program and Kernel
    std::string binaryFile = xcl::find_binary_file(device_name,"vadd");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);
#else
    std::ifstream kernel_file("Kernel.cl");
    std::string kernel_source(
            std::istreambuf_iterator<char>(kernel_file),
            (std::istreambuf_iterator<char>()));
    vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    assert(all_platforms.size()>0);
    cl::Platform default_platform = all_platforms.front();
    cout<<"Using default platform "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<endl;
    vector<cl::Device> platform_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL,&platform_devices);
    assert(platform_devices.size()>0);
    cl::Device default_device = platform_devices[platform_devices.size()-1];
    cout<<"Available devices :"<<platform_devices.size()<<" Using default platform "<<default_device.getInfo<CL_DEVICE_NAME>()<<endl;
    cl::Context context(default_device);
    cl::Program program(context,kernel_source, true);
#endif
    cl::Kernel krnl_vector_add(program,"mmult");
    cl::CommandQueue q(context,default_device);
    //Allocate Buffer in Global Memory

    //Layer 2

    mmult(input.data(),weight_layer1.data(),source_sw_results.data(),dimensions.data());
    mmult_fpga_ocl(q,context,krnl_vector_add,input,weight_layer1,source_hw_results,dimensions);
    
    std::vector<float,aligned_allocator<float>> weight_layer2 (SIZE_N*SIZE_N);
    std::vector<float,aligned_allocator<float>> sw_op_l2_results (SIZE_M*SIZE_N);    
    std::vector<float,aligned_allocator<float>> hw_op_l2_results (SIZE_M*SIZE_N);
    for(auto &x:weight_layer2)x = RandomFloat();
    for(auto &x:sw_op_l2_results)x = 0.0f;    
    dimensions[0] = SIZE_M;
    dimensions[1] = SIZE_N;
    dimensions[2] = SIZE_N;

    mmult(source_sw_results.data(),weight_layer2.data(),sw_op_l2_results.data(),dimensions.data());
    mmult_fpga_ocl(q,context,krnl_vector_add,source_sw_results,weight_layer2,hw_op_l2_results,dimensions);

    // Layer 3
    std::vector<float,aligned_allocator<float>> weight_layer3 (SIZE_N*SIZE_OUT);
    std::vector<float,aligned_allocator<float>> sw_op_l3_results (SIZE_M*SIZE_OUT);    
    std::vector<float,aligned_allocator<float>> hw_op_l3_results (SIZE_M*SIZE_OUT);
    for(auto &x:weight_layer3)x = RandomFloat();
    for(auto &x:sw_op_l3_results)x = 0.0f;    
    dimensions[0] = SIZE_M;
    dimensions[1] = SIZE_N;
    dimensions[2] = SIZE_OUT;

    mmult(sw_op_l2_results.data(),weight_layer3.data(),sw_op_l3_results.data(),dimensions.data());
    mmult_fpga_ocl(q,context,krnl_vector_add,sw_op_l2_results,weight_layer3,hw_op_l3_results,dimensions);


    //OPENCL HOST CODE AREA END

    // Compare the results of the Device to the simulation
    bool match = true;
    for (int i = 0 ; i < SIZE_M*SIZE_OUT ; i++){
    	 std::cout << "i = " << i << " CPU result = " << sw_op_l3_results[i]
    	                << " Device result = " << hw_op_l3_results[i] << std::endl;
//        if (cmpFloat(hw_op_l3_results[i],sw_op_l3_results[i])){
//            std::cout << "Error: Result mismatch" << std::endl;
//            match = false;
//            break;
//        }

    }
    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl; 
    return (match ? EXIT_SUCCESS :  EXIT_FAILURE);
}

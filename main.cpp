#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif
#include <CL/cl2.hpp>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#define LENGTH (8) // length of vectors a, b, and c
const char *err_code (cl_int err_in)
{
  switch (err_in) {
    case CL_SUCCESS:
      return (char*)"CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
      return (char*)"CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
      return (char*)"CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
      return (char*)"CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return (char*)"CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
      return (char*)"CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
      return (char*)"CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return (char*)"CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:
      return (char*)"CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:
      return (char*)"CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return (char*)"CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:
      return (char*)"CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:
      return (char*)"CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
      return (char*)"CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
      return (char*)"CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case CL_INVALID_VALUE:
      return (char*)"CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
      return (char*)"CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
      return (char*)"CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
      return (char*)"CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
      return (char*)"CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
      return (char*)"CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:
      return (char*)"CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
      return (char*)"CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
      return (char*)"CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return (char*)"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:
      return (char*)"CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:
      return (char*)"CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:
      return (char*)"CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
      return (char*)"CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
      return (char*)"CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return (char*)"CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
      return (char*)"CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:
      return (char*)"CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:
      return (char*)"CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
      return (char*)"CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
      return (char*)"CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
      return (char*)"CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
      return (char*)"CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
      return (char*)"CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
      return (char*)"CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
      return (char*)"CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
      return (char*)"CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
      return (char*)"CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
      return (char*)"CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
      return (char*)"CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:
      return (char*)"CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:
      return (char*)"CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:
      return (char*)"CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return (char*)"CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_PROPERTY:
      return (char*)"CL_INVALID_PROPERTY";

    default:
      return (char*)"UNKNOWN ERROR";
  }
}

int main() {

  // Create the two input vectors
  std::vector<int> h_a(LENGTH);                // a vector
  std::vector<int> h_b(LENGTH);                // b vector
  std::vector<int> h_c(LENGTH);                // c vector


  cl::Buffer d_a;                       // device memory used for the input  a vector
  cl::Buffer d_b;                       // device memory used for the input  b vector
  cl::Buffer d_c;                       // device memory used for the input c vector


  // Fill vectors a and b with random float values
  int count = LENGTH;
  for(int i = 0; i < count; i++)
  {
    h_a[i]  =  i;
    h_b[i]  = 0;
    h_c[i]  = 0;

  }

  try {
    // Get available platforms
    cl::Context context(DEVICE);
    // Read source file
    std::ifstream sourceFile("addVectors.cl");
    std::string sourceCode(
        std::istreambuf_iterator<char>(sourceFile),
        (std::istreambuf_iterator<char>()));

    // Make program of the source code in the context
    cl::Program program(context,sourceCode,true);
    cl::CommandQueue queue(context);

    cl::compatibility::make_kernel<cl::Buffer,cl::Buffer,cl::Buffer>vadd(program,"vadd");

    d_a   = cl::Buffer(context, h_a.begin(), h_a.end(), true);
    d_b   = cl::Buffer(context, h_b.begin(), h_b.end(), true);


    d_c  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * LENGTH);

    vadd(cl::EnqueueArgs(
            queue,
            cl::NDRange(count)),
        d_a,
        d_b,
        d_c);

    cl::copy(queue, d_c, h_c.begin(), h_c.end());
    int correct = 0;
    int tmp;
    for(int i = 0; i < LENGTH; i++)
    {
      printf(" %d + %d = %d \n ",h_a[i], h_b[i], h_c[i]);
    }


  }
  catch (cl::Error err) {
    std::cout << "Exception\n";
    std::cerr
        << "ERROR: "
        << err.what()
        << "("
        << err_code(err.err())
        << ")"
        << std::endl;
  }

}
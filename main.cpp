#include <iostream>
#include <CL/cl2.hpp>
#include <fstream>
#include <cassert>
#define LOAD 4096
using std::cout;
using std::endl;

int main () {
  cl_int err;
  
  std::srand(std::time(nullptr));
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get (&all_platforms);
  std::vector<cl::Device> devices;
  for(auto x:all_platforms)cout<<"Platforms: "<<x.getInfo <CL_PLATFORM_NAME>()<<endl;
  all_platforms.front ().getDevices (CL_DEVICE_TYPE_ALL,&devices);
  for(auto x:devices)cout<<"Devices: "<<x.getInfo <CL_DEVICE_NAME>()<<endl;
  cl::Device default_device = devices.front ();
  cl::Context context(default_device);
  cl::CommandQueue queue(context,default_device);
  
  std::ifstream kernel_file ("kernel.cl");
  std::string kernel_source (
      std::istreambuf_iterator<char> (kernel_file) ,
      ( std::istreambuf_iterator<char> ()));
  
  cl::Program program(context,kernel_source);
  err = program.build ("-g");
  
  assert(err==CL_BUILD_SUCCESS);
  cl::Kernel vadd(program,"vadd",&err);
  assert(err==CL_SUCCESS);
  std::vector<int> A = std::vector<int>(LOAD,1);
  std::vector<int> B = std::vector<int>(LOAD);  
  std::vector<int> C = std::vector<int>(LOAD,0);
  std::vector<int> C_sim = std::vector<int>(LOAD,0);
  int block_size;
  
  for(auto &x:B)x = rand()%20;
  for(int i =0;i<LOAD;i++){
    C_sim[i]=A[i]+B[i];
  }
  
  cl::Buffer buffer_A(context,CL_MEM_READ_ONLY, sizeof (int)*LOAD,NULL,&err);
  cl::Buffer buffer_B(context,CL_MEM_READ_ONLY, sizeof (int)*LOAD,NULL,&err);
  cl::Buffer buffer_C(context,CL_MEM_WRITE_ONLY, sizeof (int)*LOAD,NULL,&err);
  cl::Buffer buffer_block(context,CL_MEM_READ_ONLY, sizeof (int),NULL,&err);

  int nargs = 0;
  vadd.setArg (nargs++,buffer_A);
  vadd.setArg (nargs++,buffer_B);
  vadd.setArg (nargs++,buffer_C);
  vadd.setArg (nargs,buffer_block);
  
  queue.enqueueWriteBuffer (buffer_A,CL_TRUE,0, sizeof (int)*LOAD,A.data ());
  queue.enqueueWriteBuffer (buffer_B,CL_TRUE,0, sizeof (int)*LOAD,B.data ());
  queue.enqueueWriteBuffer (buffer_block,CL_TRUE,0, sizeof (int),&block_size);

  cl::Event event;
  queue.enqueueNDRangeKernel (vadd, 0, cl::NDRange(4), cl::NullRange, NULL, &event);
  event.wait ();

  queue.enqueueReadBuffer (buffer_C,CL_TRUE,0, sizeof (int)*LOAD,C.data ());
  queue.finish ();
  bool match = true;


  return match?EXIT_SUCCESS:EXIT_FAILURE;
}
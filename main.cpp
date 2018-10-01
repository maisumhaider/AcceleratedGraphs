#include <iostream>
#include <CL/cl.hpp>
#include <cassert>
#include <fstream>
#include <ctime>

#define SIZE 64
using std::cout;
using std::vector;
using std::endl;
void multiplyMatrices(const int *matA,const int *matB,int *matC,const int dim) {
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      int sum = 0;
      for (int k = 0; k < dim; k++)
        sum = sum + matA[i * dim+ k] * matB[k * dim + j];
      matC[i * dim + j] = sum;
    }

  }
}
int main(){
  srand (time(NULL));
  cl_int res;
  int A[SIZE*SIZE];
  int B[SIZE*SIZE];
  int C_SIM[SIZE*SIZE];
  int C[SIZE*SIZE];
  int dim[1] = {SIZE};

  for(int i=0;i<(SIZE*SIZE);i++){
    A[i] = rand()% SIZE;
    B[i] = rand()% SIZE;
    C_SIM[i] = 0;
    C[i] = 0;
  }
  multiplyMatrices(A,B,C_SIM,SIZE);
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


  cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * SIZE*SIZE);
  cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * SIZE*SIZE);
  cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * SIZE*SIZE);
  cl::Buffer buffer_N(context, CL_MEM_READ_WRITE, sizeof(int));

  int nargs=0;
  cl::Kernel mmult(program,"mmult");
  mmult.setArg(nargs++,buffer_A);
  mmult.setArg(nargs++,buffer_B);
  mmult.setArg(nargs++,buffer_C);
  mmult.setArg(nargs,buffer_N);

  cl::CommandQueue queue(context,default_device);
  queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(int) * SIZE*SIZE,A);
  queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,sizeof(int) * SIZE*SIZE,B);
  queue.enqueueWriteBuffer(buffer_N,CL_TRUE,0,sizeof(int),dim);
  queue.enqueueNDRangeKernel(mmult,cl::NullRange,cl::NDRange(1),cl::NullRange);
  queue.finish();
  queue.enqueueReadBuffer(buffer_C,CL_TRUE,0,sizeof(int) * SIZE*SIZE,C);

  for(int i=0;i<SIZE*SIZE;i++){
    assert(C[i]==C_SIM[i]);
  }
  cout<<"All tested passed"<<endl;
  cout<<endl;
  return 0;
}

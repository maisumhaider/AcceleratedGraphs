#include <iostream>
#include <CL/cl2.hpp>
#include <fstream>
#include <cassert>
#include <cmath>
#define LOAD 4096
#define CORES 4
using std::cout;
using std::endl;

template <typename T>
struct aligned_allocator {
  using value_type = T;
  T *allocate (std::size_t num) {
    void *ptr = nullptr;
    if ( posix_memalign (&ptr , 4096 , num * sizeof (T))) {
      throw std::bad_alloc ();
    }
    return reinterpret_cast<T *>(ptr);
  }
  void deallocate (T *p , std::size_t num) {
    free (p);
  }
};
bool cmpFloat (float A , float B , float tolerance = 0.0f);
float RandomFloat (float min = 0.0 , float max = 1.0);

void mmult(const int id, const float* matA, const float* matB,
           float* matC, const int* dimM, const int* dimK, const int* dimN, const int* block_size);

void mmult_OpenCL_coalesced(cl::CommandQueue& q,cl::Context &context,cl::Kernel &kernel);

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
  int global = 4;
  int block_size=LOAD/global;

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
  queue.enqueueNDRangeKernel (vadd, 0, cl::NDRange(global), cl::NullRange, NULL, &event);
  event.wait ();

  queue.enqueueReadBuffer (buffer_C,CL_TRUE,0, sizeof (int)*LOAD,C.data ());
  queue.finish ();
  bool match = true;
  for(int i =0;i<LOAD;i++){
    if (C_sim[i]!=C[i]){
      cout<<"Id:"<<i<<" SW Res:"<<C_sim[i]<<" HW Res:"<<C[i]<<endl;
      match= false;
      break;
    }
  }

  return match?EXIT_SUCCESS:EXIT_FAILURE;
}
void mmult (const int id ,
            const float *matA ,
            const float *matB ,
            float *matC ,
            const int *dimM ,
            const int *dimK ,
            const int *dimN ,
            const int *block_size) {
  int k,i,startRow,startCol;
  int index,index2,resIndex;
  float acc = 0.0f;
  startRow = (id * block_size[0])/dimM[0];
  startCol = (id * block_size[0])%dimM[0];
  printf("Executing Kernel\n");
  for(i=0;i<block_size[0];i++) {
    for ( k = 0 ; k < dimK[0] ; k++ ) {
      index = startRow * dimK[0] + k;
      index2 = startCol* dimK[0] + k;
      acc += matA[ index ] * matB[ index2 ];
    }
    resIndex = startRow * dimN[0] + startCol;
    matC[ resIndex ] = acc;
  }
}
bool cmpFloat (float A , float B , float tolerance) {
  if ( fabs (A - B) > std::numeric_limits<float>::epsilon ()) {
    cout << "Mismatch " << fabs (A - B) << endl;
    return false;
  }
  return true;
}
float RandomFloat (float min , float max) {
  assert(max > min);
  float random = (( float ) rand ()) / ( float ) RAND_MAX;
  float range = max - min;
  return ( random * range ) + min;
}

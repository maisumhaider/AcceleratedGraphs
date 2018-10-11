#include <iostream>
#include <iterator>
#include <CL/cl.hpp>
#include <cassert>
#include <fstream>
#include <ctime>
#include <cmath>

#define SIZE 1
#define SIZE2 784
#define SIZE3 500
#define OUTPUTSIZE 10
using std::cout;
using std::vector;
using std::endl;


bool cmpf(float A, float B, float tolerance = 0.05f) {
    return (fabs(A - B) < tolerance);
}


float RandomFloat(float min = 0.0, float max = float(SIZE))
{
    // this  function assumes max > min, you may want
    // more robust error checking for a non-debug build
    assert(max > min);
    float random = ((float) rand()) / (float) RAND_MAX;

    // generate (in your case) a float between 0 and (4.5-.78)
    // then add .78, giving you a float between .78 and 4.5
    float range = max - min;
    return (random*range) + min;
}
float relu(float input){
    return input > 0 ? input:0;
}
void multiplyMatrices(const float *matA,const float *matB,float *matC,const int *dim) {
  int dimM = dim[0],dimN= dim[2],dimK= dim[1];

  for (int m=0; m<dimM; m++) {
    for (int n=0; n<dimN; n++) {
      float acc = 0.0f;
      for (int k=0; k<dimK; k++) {
        acc += matA[k*dimM + m] * matB[n*dimK + k];
      }
      matC[n*dimM + m] = relu(acc);
    }
  }
}
void setupArray(float *arr, int size, bool setRandom= false, float max=1.0){
    for(int i =0 ; i< size; i++){
                arr[i] = setRandom ? RandomFloat(0,max) : max;
    }
}
void mmult_FPGA(const float* input,const int input_dim, const float* weight_layer1,const int weight_dim, float* output,const int output_dim, const int* dim_layer1)
{
    std::ifstream kernel_file("Kernel.cl");
    std::__cxx11::string kernel_source(
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

    cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(float) * input_dim);
    cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(float) * weight_dim);
    cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(float) * output_dim);
    cl::Buffer buffer_N(context, CL_MEM_READ_WRITE, sizeof(int)*3);

    int nargs=0;
    cl::Kernel mmult(program,"mmult");
    mmult.setArg(nargs++,buffer_A);
    mmult.setArg(nargs++,buffer_B);
    mmult.setArg(nargs++,buffer_C);
    mmult.setArg(nargs,buffer_N);

    cl::CommandQueue queue(context,default_device);
    queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(float) * input_dim,input);
    queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,sizeof(float) * weight_dim,weight_layer1);
    queue.enqueueWriteBuffer(buffer_N,CL_TRUE,0,sizeof(int)*3,dim_layer1);
    queue.enqueueNDRangeKernel(mmult, cl::NullRange, cl::NDRange(1), cl::NullRange);
    queue.finish();
    queue.enqueueReadBuffer(buffer_C,CL_TRUE,0,sizeof(float) * output_dim,output);
    queue.finish();
}
void mlp_compute(float *input, float *weight,float *output, int *dims){
    int input_dim = dims[0]*dims[1];
    int weight_dim = dims[1]*dims[2];
    int output_dim = dims[0]*dims[2];
    float* out_sim = new float[output_dim];

    setupArray(input,input_dim, true,SIZE2);

    setupArray(weight,weight_dim, true,1.0);
    setupArray(out_sim,output_dim, false,0.0);
    setupArray(output,output_dim, false,0.0);

    multiplyMatrices(input,weight,out_sim,dims);
    mmult_FPGA(input,input_dim, weight,weight_dim, output,output_dim, dims);

    for(int i=0;i<output_dim;i++){
//        assert(cmpf(output[i],out_sim[i]));
            if( not cmpf(output[i],out_sim[i])) {
                cout << output[i] << " " << out_sim[i] << endl;
                exit(-1);
            }
    }
    delete[] out_sim;

}
int main(){
    srand (time(NULL));
    int *dim_layer = new int[3]{SIZE,SIZE2,SIZE3};

    float* input = new float[dim_layer[0]*dim_layer[1]];
    float* weight_layer = new float[dim_layer[1]*dim_layer[2]];
    float* mlp_layer_output = new float[dim_layer[0]*dim_layer[2]];

    cout<<"Computing First MLP Layer 1 "<<endl;
    mlp_compute(input,weight_layer,mlp_layer_output,dim_layer);
    delete[] weight_layer;
    delete[] input;
    delete[] dim_layer;
    dim_layer = new int[3]{SIZE,SIZE3,SIZE3};
    weight_layer = new float[dim_layer[1]*dim_layer[2]];
    float* mlp_layer2_output = new float[dim_layer[0]*dim_layer[2]];

    cout<<"Computing First MLP Layer 2 "<<endl;
    mlp_compute(mlp_layer_output,weight_layer,mlp_layer2_output,dim_layer);
    delete[] weight_layer;
    delete[] mlp_layer_output;
    delete[] dim_layer;
    dim_layer = new int[3]{SIZE,SIZE3,OUTPUTSIZE};
    weight_layer = new float[dim_layer[1]*dim_layer[2]];
    float* mlp_layer3_output = new float[dim_layer[0]*dim_layer[2]];
    cout<<"Computing First MLP Layer 3"<<endl;
    mlp_compute(mlp_layer2_output,weight_layer,mlp_layer3_output,dim_layer);

    cout<<"All tested passed"<<endl;
    cout<<endl;
    return 0;
}

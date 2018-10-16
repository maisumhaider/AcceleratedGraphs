__kernel void vadd(__global const int* A, __global const int* B, __global int* C, __global const int* block_size){
  int id = get_global_id(0);
  printf("Executing Kernel %d\n",id);
  int i;
  int index = id*block_size[0];
  for(i=0;i<block_size[0];i++){
    C[index+i] = A[index+i]+B[index+i];
  }
}
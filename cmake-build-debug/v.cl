__kernel void vadd(__global const int* A, __global const int* B, __global int* C, const int block_size){
  int id = get_global_id(0);
  printf("Executing Kernel %d\n",id);
  int i;
  int index = id*block_size;
  for(i=0;i<block_size;i++){
    C[index+i] = A[index+i]+B[index+i];
  }
}
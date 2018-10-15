__kernel void vadd(__global const int* A, __global const int* B, __global int* C, __global const int* block_size){
  int id = get_global_id(0);
  printf("Executing Kernel %d\n",id);
  int i;
  for(i=0;i<block_size[0];i++){
    C[id+i] = A[id+i]+B[id+i];
  }
}
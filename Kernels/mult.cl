kernel void mult(__global const int *A,__global const int *B,__global int *C){
  uint id = get_global_id(0);
  C[id] = A[id]*B[id];
}

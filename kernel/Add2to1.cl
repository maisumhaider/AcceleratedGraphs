__kernel void Add2to1(__global const int* A, __global const int* B, __global int* C ) {
  int ID;
  ID = get_local_id(0);
  C[ID] = A[ID] + B[ID];
}


__kernel void Inc2(__global const int* A, __global const int* B ) {
  int ID;
  ID = get_local_id(0);
  B[ID] = A[ID] + 2;
}

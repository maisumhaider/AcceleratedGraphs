
__kernel void Mul23(__global const int* A, __global const int* B, __global const int* C ) {
  int ID;
  ID = get_local_id(0);
  int x = A[ID];
  B[ID] = x * 2;
  C[ID] = x * 3;
}


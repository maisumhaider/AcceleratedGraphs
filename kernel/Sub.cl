
__kernel void Sub2(__global const int* A, __global const int* B ) {
  int ID;
  ID = get_local_id(0);
  B[ID] = A[ID] - 2;
}


__kernel void Sub3(__global const int* A, __global const int* B ) {
  int ID;
  ID = get_local_id(0);
  B[ID] = A[ID] - 3;
}

__kernel void vadd(__global const int* A, __global const int* B, __global int* C, __global const int* block_size){
  int id = get_global_id(0);
  printf("Executing Kernel %d\n",id);
  int i;
  int index = id*block_size[0];
  for(i=0;i<block_size[0];i++){
    C[index+i] = A[index+i]+B[index+i];
  }
}

__kernel void mmult(__global const float* matA,  //Read-only input matrix1
                    __global const float* matB,  //Read-only input matrix2
                    __global float* matC,  //Output matrix
                    __global const int* dimM,
                    __global const int* dimK,
                    __global const int* dimN,
                    __global const int* block_size){
  int id = get_global_id(0);//id/dimM;
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
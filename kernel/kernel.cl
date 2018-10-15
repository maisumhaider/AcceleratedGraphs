kernel void mmult( __global float* matA,  //Read-only input matrix1
                     __global float* matB,  //Read-only input matrix2
                     __global float* matC,  //Output matrix
                     __global int* max             //One dimension of the matrix
)
{
    int globalRow = get_global_id(0);
    int globalCol = get_global_id(1);
    int dimM = max[0];
    int dimK =  max[1];
    int dimN = max[2];
    int m,n,k;
    int index,index2,resIndex;
//    for (m=0; m<dimM; m++) {
//        for (n=0; n<dimN; n++) {
            float acc = 0.0f;
            for (k=0; k<dimK; k++) {
                index = m*dimK+k;
                index2 = n*dimK+k;
                acc += matA[index]*matB[index2];
            }
            resIndex = m*dimN +n;
            matC[resIndex] = acc;
//        }
//    }
}
kernel void mmult( __global float* matA,  //Read-only input matrix1
                     __global float* matB,  //Read-only input matrix2
                     __global float* matC,  //Output matrix
                     __global int* max             //One dimension of the matrix
)
{
    int id = get_global_id(0);
    int dimM = max[id];
    int dimK =  max[id+1];
    int dimN = max[id+2];
    int m,n,k;
    for (m=0; m<dimM; m++) {
        for (n=0; n<dimN; n++) {
            float acc = 0.0f;
            for (k=0; k<dimK; k++) {
                acc += matA[k*dimM + m] * matB[n*dimK + k];
            }
            matC[n*dimM + m] = acc > 0 ? acc:0;
        }
    }
}
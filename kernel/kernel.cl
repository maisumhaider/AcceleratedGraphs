__kernel void simple_add(__global const int* A, __global const int* B,__global int* C){
    int  id = get_global_id(0);
    C[id]=A[id]+B[id];
}
__kernel void mmult( __global int* matA,  //Read-only input matrix1
                     __global int* matB,  //Read-only input matrix2
                     __global int* matC,  //Output matrix
                     __global int* max             //One dimension of the matrix
)
{
    int dim = max[get_global_id(0)];
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            int sum = 0;
            for (int k = 0; k < dim; k++)
                sum = sum + matA[i * dim+ k] * matB[k * dim + j];
            matC[i * dim + j] = sum;
        }

    }
}
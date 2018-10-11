__kernel void simple_add(__global const int* A, __global const int* B,__global int* C){
    int  id = get_global_id(0);
    C[id]=A[id]+B[id];
}

float relu(float input){
    return input > 0 ? input:0;
}
__kernel void mmult( __global float* matA,  //Read-only input matrix1
                     __global float* matB,  //Read-only input matrix2
                     __global float* matC,  //Output matrix
                     __global int* max             //One dimension of the matrix
)
{
    int id = get_global_id(0);
    int dimM = max[id];
    int dimK=  max[id+1];
    int dimN= max[id+2];
//    printf("\nExecuting Kernel.\n\n");
//    int iter;
//    printf("Printing matrix A.\n");
//    for(iter = 0;iter<dimM*dimK;iter++){
//        printf("%f ",matA[iter]);
//    }
//    printf("\nPrinting matrix B.\n");
//    for(iter = 0;iter<dimN*dimK;iter++){
//        printf("%f ",matB[iter]);
//    }
//    printf("\n");
    for (int m=0; m<dimM; m++) {
        for (int n=0; n<dimN; n++) {
            float acc = 0.0f;
            for (int k=0; k<dimK; k++) {
                acc += matA[k*dimM + m] * matB[n*dimK + k];
            }
            matC[n*dimM + m] = relu(acc);
        }
    }
}
//
// Created by olibo on 07/11/2022.
//

#include <stdio.h>

#include <cuda.h>

__global__ void testMultiplyKernel(int *a, int *b, int *out){
    int tid = blockIdx.x*blockDim.x+threadIdx.x;

    out[tid] = a[tid]*b[tid];
}

int testMultiply(){
    int N = 10;

    int* h_a;
    int* h_b;
    int* h_out;

    int* d_a;
    int* d_b;
    int* d_out;

    h_a = (int*)malloc(sizeof(int)*N);
    h_b = (int*)malloc(sizeof(int)*N);
    h_out = (int*)malloc(sizeof(int)*N);

    // Initialise inputs
    int i;
    for(i = 0; i < N; i++){
        h_a[i] = i;
        h_b[i] = i;
    }

    cudaMalloc((void**)&d_a, sizeof(int)*N);
    cudaMalloc((void**)&d_b, sizeof(int)*N);
    cudaMalloc((void**)&d_out, sizeof(int)*N);

    cudaMemcpy(d_a, h_a, sizeof(int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b , sizeof(int)*N, cudaMemcpyHostToDevice);

    testMultiplyKernel<<<1, N>>>(d_a, d_b, d_out);

    cudaMemcpy(h_out, d_out, sizeof(int)*N, cudaMemcpyDeviceToHost);

    for(i = 0; i < N; i++)
    {
        printf("%i * %i = %i \n", h_a[i], h_b[i], h_out[i]);
    }

    return 0;
}
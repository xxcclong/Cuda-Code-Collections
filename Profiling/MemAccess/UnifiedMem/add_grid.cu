#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <math.h>
 
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
      if (result != cudaSuccess) {
              fprintf(stderr, "CUDA Runtime Error: %s\n",
                                  cudaGetErrorString(result));
                  assert(result == cudaSuccess);
                    }
#endif
        return result;
}
// CUDA kernel to add elements of two arrays
__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}
 
int main(void)
{
    int N = 1<<20;
    float *x, *y;

    // Allocate Unified Memory -- accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Launch kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    
    cudaEvent_t startEvent, endEvent;
    checkCuda( cudaEventCreate(&startEvent));
    checkCuda( cudaEventCreate(&endEvent));
    checkCuda( cudaEventRecord(startEvent, 0));

    add<<<numBlocks, blockSize>>>(N, x, y);

    checkCuda(cudaEventRecord(endEvent, 0));
    checkCuda(cudaEventSynchronize(endEvent));
    float ti;
    checkCuda(cudaEventElapsedTime(&ti, startEvent, endEvent));
    printf("used time %f\n", ti);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y) ;

    return 0;
}

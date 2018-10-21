
#include <stdio.h>
#include <assert.h> 
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
__global__ void staticReverse(int *d, int n)
{
  __shared__ int s[64];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

__global__ void dynamicReverse(int *d, int n)
{
  extern __shared__ int s[];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

int main(void)
{
  cudaEvent_t startEvent, endEvent;
  checkCuda( cudaEventCreate(&startEvent));
  checkCuda( cudaEventCreate(&endEvent));
  const int n = 64;
  int a[n], r[n], d[n];
  
  for (int i = 0; i < n; i++) {
    a[i] = i;
    r[i] = n-i-1;
    d[i] = 0;
  }

  int *d_d;
  cudaMalloc(&d_d, n * sizeof(int)); 
  
  // run version with static shared memory
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  checkCuda( cudaEventRecord(startEvent, 0));
  staticReverse<<<1,n>>>(d_d, n);
  checkCuda(cudaEventRecord(endEvent, 0));
  checkCuda(cudaEventSynchronize(endEvent));
  float ti;
  checkCuda(cudaEventElapsedTime(&ti, startEvent, endEvent));
  printf("used time %f\n", ti);

  cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
  
  // run dynamic shared memory version
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  checkCuda( cudaEventRecord(startEvent, 0));
  dynamicReverse<<<1,n,n*sizeof(int)>>>(d_d, n);

  checkCuda(cudaEventRecord(endEvent, 0));
  checkCuda(cudaEventSynchronize(endEvent));
  checkCuda(cudaEventElapsedTime(&ti, startEvent, endEvent));
  printf("used time %f\n", ti);

  cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
}

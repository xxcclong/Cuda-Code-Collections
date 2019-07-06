/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory>
#include <vector>
#include <thread>

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nccl.h>

/* Matrix size */
#define N (256 * 4)
//#define N (256*8)
#define GPUS (2)
#define ITERATIONS (200)
//#define ITERATIONS (10)
#define COMPUTE_TIME 5
#define COMM_TIME 2
//float *h_C_ref;
float *d_A[GPUS];
float *d_B[GPUS];
float *d_C[GPUS];
float *d_D[GPUS];
float alpha = 1.0f;
float beta = 0.0f;
int n2 = N * N;
int i;

enum NCCL_MODE {
  ASYNC = 0,
  SYNC = 1,
  ONE_STREAM = 2,
  NO_COMM = 3,
  NO_COMPUTE = 4
};


std::unique_ptr<ncclComm_t[]> comms = nullptr;
std::unique_ptr<cudaStream_t[]> nccl_streams = nullptr;
std::unique_ptr<cudaStream_t[]> blas_streams = nullptr;
size_t timestamp() {
  using namespace std::chrono;
  return duration_cast<microseconds>(
    high_resolution_clock::now().time_since_epoch()).count();
}

void init_nccl() {
  comms.reset(new ncclComm_t[2]);
  nccl_streams.reset(new cudaStream_t[2]);
  blas_streams.reset(new cudaStream_t[2]);
  ncclUniqueId nccl_id;
  ncclGetUniqueId(&nccl_id);
  ncclGroupStart();
  for (size_t i = 0; i < GPUS; ++i) {
    cudaSetDevice(i);
    cudaStreamCreate(nccl_streams.get()+i);
    ncclCommInitRank(comms.get() + i, GPUS, nccl_id, i);
    cudaStreamCreate(blas_streams.get()+i);
  }
  ncclGroupEnd();
}

int init_data(int dev) {
  float *ha;
  float *hb;
  float *hc;
  //float *h_C_ref;
  d_A[dev] = 0;
  d_B[dev] = 0;
  d_C[dev] = 0;
  d_D[dev] = 0;
  //float *da = *d_A[dev] = 0;
  //float *db = *d_B[dev] = 0;
  //float *dc = *d_C[dev] = 0;
  cudaSetDevice(dev);
  /* Allocate host memory for the matrices */
  ha = reinterpret_cast<float *>(malloc(n2 * sizeof(ha[0])));
  hb = reinterpret_cast<float *>(malloc(n2 * sizeof(hb[0])));
  hc = reinterpret_cast<float *>(malloc(n2 * sizeof(hc[0])));

  /* Fill the matrices with test data */
  float e = rand() / static_cast<float>(RAND_MAX);
  for (i = 0; i < n2; i++) {
    ha[i] = hb[i] = hc[i] = e;
  }

  /* Allocate device memory for the matrices */
  if (cudaMalloc(reinterpret_cast<void **>(&d_A[dev]), n2 * sizeof(d_A[dev][0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
    return EXIT_FAILURE;
  }

  if (cudaMalloc(reinterpret_cast<void **>(&d_B[dev]), n2 * sizeof(d_B[dev][0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
    return EXIT_FAILURE;
  }

  if (cudaMalloc(reinterpret_cast<void **>(&d_C[dev]), n2 * sizeof(d_C[dev][0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
    return EXIT_FAILURE;
  }

  if (cudaMalloc(reinterpret_cast<void **>(&d_D[dev]), n2 * sizeof(d_D[dev][0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate D)\n");
    return EXIT_FAILURE;
  }


  /* Initialize the device matrices with the host matrices */
  cublasSetVector(n2, sizeof(ha[0]), ha, 1, d_A[dev], 1);
  cublasSetVector(n2, sizeof(hb[0]), hb, 1, d_B[dev], 1);
  cublasSetVector(n2, sizeof(hc[0]), hc, 1, d_C[dev], 1);
  cublasSetVector(n2, sizeof(hc[0]), hc, 1, d_D[dev], 1);
  return 0;
}

int destroy_data(int dev) {
  //float *h_C_ref;
  float *da = d_A[dev];
  float *db = d_B[dev];
  float *dc = d_C[dev] ;
  float *dd = d_D[dev] ;
  /* Memory clean up */

  if (cudaFree(da) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (A)\n");
    return EXIT_FAILURE;
  }

  if (cudaFree(db) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (B)\n");
    return EXIT_FAILURE;
  }

  if (cudaFree(dc) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
  }

  if (cudaFree(dd) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
  }
  return 0;
}

/* Main */
int worker(int dev, int nccl_mode) {
  cublasStatus_t status;

  cublasHandle_t handle;
  auto &blas_stream = *(blas_streams.get() + dev);
  cudaSetDevice(dev);

  status = cublasCreate(&handle);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }

  cublasSetStream(handle, blas_stream);

  /* Performs operation using cublas */
  auto &nccl_stream = *(nccl_streams.get() + dev);
  std::vector<cudaEvent_t> nccl_events;
  nccl_events.reserve(ITERATIONS);
  size_t start = timestamp();
  if (nccl_mode == NCCL_MODE::ONE_STREAM) {
    for (size_t i = 0; i <ITERATIONS; ++i) {
      for(int temp = 0; temp < COMPUTE_TIME; ++temp)
          status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A[dev],
                            N, d_B[dev], N, &beta, d_C[dev], N);
      for(int temp = 0; temp < COMM_TIME; ++temp)
      ncclAllReduce(d_D[dev], d_D[dev], n2, ncclFloat, ncclSum, *(comms.get() + dev), blas_stream);
    }
    cudaStreamSynchronize(blas_stream);
  } else {
    // nccl_mode is ASYNC_NCCL or SYNC_NCCL
    for (size_t i = 0; i < ITERATIONS; ++i) {
      if(nccl_mode != NO_COMPUTE){
          for(int temp = 0; temp < COMPUTE_TIME; ++temp)
              status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A[dev],
                                N, d_B[dev], N, &beta, d_C[dev], N);
        }



      if(nccl_mode != NO_COMM)
          for(int temp = 0; temp < COMM_TIME; ++temp){
              ncclAllReduce(d_D[dev], d_D[dev], n2, ncclFloat, ncclSum, *(comms.get() + dev), nccl_stream);
              if (nccl_mode == SYNC) {
                    cudaStreamSynchronize(nccl_stream);
                  }
          }
      else
          for(int temp = 0; temp < COMM_TIME; ++temp){
              ncclAllReduce(d_D[dev], d_D[dev], 1, ncclFloat, ncclSum, *(comms.get() + dev), nccl_stream);
              if (nccl_mode == SYNC) {
                    cudaStreamSynchronize(nccl_stream);
                  }
          }
          
      
    }
    cudaStreamSynchronize(nccl_stream);
    cudaStreamSynchronize(blas_stream);
  }
  fprintf(stderr, "device: [%d], %d iterations spent: [%d ms]\n", dev, ITERATIONS, (timestamp()-start)/1000);


  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! kernel execution error.\n");
    return EXIT_FAILURE;
  }

  /* Shutdown */
  status = cublasDestroy(handle);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! shutdown error (A)\n");
    return EXIT_FAILURE;
  }
  return 0;
}

int main(int argc, char** argv) {
  if(argc != 2)
  {
    printf("USAGE: ./a.out 2 # 0 sync, 1 async, 2 one stream\n");
    return -1;
  }

  init_nccl();
  for (int i = 0; i < GPUS; ++i) {
    init_data(i);
  }
  std::vector<std::thread> threads;
  size_t start = timestamp();
  int nccl_mode = atoi(argv[1]);
  printf("nccl mode %d\n", nccl_mode);
  for (int i = 0; i < GPUS; ++i) {
    std::thread t(std::bind(&worker, i, nccl_mode));
    threads.push_back(std::move(t));
  }
  for (auto &t : threads) {
    t.join();
  }
  fprintf(stderr, "nccl mode: [%d], spent: [%d ms]\n", nccl_mode, (timestamp() - start)/1000);


  for (int i = 0; i < GPUS; ++i) {
    destroy_data(i);
  }
  return 0;
}

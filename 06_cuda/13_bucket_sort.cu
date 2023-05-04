#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void init(int *key) {
  key[threadIdx.x] = 0;
}

__global__ void scan(int *a, int *b, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for(int j=1; j<N; j<<=1) {
    b[i] = a[i];
    __syncthreads();
    a[i] += b[i-j];
    __syncthreads();
  }
}

__global__ void bucketCount(int *bucket, int *key) {
  atomicAdd(&bucket[key[threadIdx.x]], 1);
}

__global__ void sort(int *key, int *bucket, int n, int range) {
  int begin = threadIdx.x == 0 ? 0 : bucket[threadIdx.x-1];
  int end = bucket[threadIdx.x];
  if (blockIdx.x >= end || blockIdx.x < begin) return;
  key[blockIdx.x] = threadIdx.x;
}

int main() {
  int n = 50;
  int range = 5;

  int *key, *bucket, *tmp;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMallocManaged(&tmp, range*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  init<<<1,range>>>(bucket);
  cudaDeviceSynchronize();

  bucketCount<<<1,n>>>(bucket, key);
  cudaDeviceSynchronize();

  scan<<<1,range>>>(bucket, tmp, range);
  cudaDeviceSynchronize();

  sort<<<n,range>>>(key, bucket, n, range);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
  cudaFree(tmp);
}


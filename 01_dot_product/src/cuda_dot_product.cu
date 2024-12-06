// cuda_dot_product.cu
#include <cuda_runtime.h>
//#include <stdio.h>

__global__ void dot_product_kernel(const float* a, const float* b, float* result, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0;

    // Use a for loop to accumulate the dot product
    for (int i = index; i < n; i += blockDim.x * gridDim.x) {
        sum += a[i] * b[i];
    }

    atomicAdd(result, sum);
}

// Ensure C linkage for the following functions
extern "C" __declspec(dllexport) void dot_product(const float* a, const float* b, float* result, int n) {
    // Function implementation...
    // Allocate device memory
    float* d_a;
    float* d_b;
    float* d_result;

    cudaMalloc((void**)& d_a, n * sizeof(float));  // Corrected line
    cudaMalloc((void**)& d_b, n * sizeof(float));
    cudaMalloc((void**)& d_result, sizeof(float));

    // Copy data to device
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize result to 0 on the device
    float zero = 0.0f;
    cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with an appropriate number of blocks and threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    dot_product_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, n);

   // Copy result back to host
   cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

   // Free device memory
   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_result);

} // End of extern "C"
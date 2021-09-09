
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t addWithCudaDouble(double* c, double* a, double* b, unsigned int size);
cudaError_t addWithCudaFloat(float* c, float* a, float* b, unsigned int size);
cudaError_t addWithCudaFloatTimes(float* c, float* a, float* b, unsigned int size, int laps);
cudaError_t addWithCudaFloatTimesBlocks(float* c, float* a, float* b, unsigned int size, int laps);


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void addKernelDouble(double* c, double* a, double* b)
{
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}

__global__ void addKernelDoubleShowIndex(double* c, double* a, double* b)
{
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
    printf("H-> [X:%d Y:%d Z:%d]   B->[X:%d Y:%d Z:%d]  {%1f + %1f}={%1f}\n", threadIdx.x, threadIdx.y, threadIdx.z
                                                                            , blockIdx.x, blockIdx.y, blockIdx.z
                                                                            , a[i], b[i], c[i]);
}

__global__ void addKernelDoubleShowIndexBlocks(double* c, double* a, double* b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] * b[i];
    printf("H-> [X:%d Y:%d Z:%d]   B->[X:%d Y:%d Z:%d]  {%1f + %1f}={%1f}\n", threadIdx.x, threadIdx.y, threadIdx.z
                                                                            , blockIdx.x, blockIdx.y, blockIdx.z
                                                                            , a[i], b[i], c[i]);
}

__global__ void multMatrixKernelFloat(float* c, float* a, float* b, int N) {
    int i = threadIdx.x;
    int j = threadIdx.y;

    float sum = 0;
    for (int k = 0; k < N; k++)
        sum += a[i * N + k] * b[k * N + j];
    c[i * N + j] = sum;

    //printf("A-> [X:%d Y:%d] [P:%d] = %f\n", i, j, (i * N + j), b[(i*N+j)]);
}

__global__ void multMatrixKernelFloatBlocks(float* c, float* a, float* b, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    float sum = 0;
    for (int k = 0; k < N; k++)
        sum += a[i * N + k] * b[k * N + j];
    c[i * N + j] = sum;

    //printf("A-> [X:%d Y:%d] [P:%d] = %f\n", i, j, (i * N + j), b[(i*N+j)]);
}

__global__ void multMatrixKernelFloatBlocks2(float* c, float* a, float* b, int N, int blockAdjust) {
    int i = blockDim.x * (blockIdx.x + blockAdjust) + threadIdx.x;
    int j = blockDim.y * (blockIdx.y + blockAdjust) + threadIdx.y;

    float sum = 0;
    for (int k = 0; k < N; k++)
        sum += a[i * N + k] * b[k * N + j];
    c[i * N + j] = sum;

    //printf("A-> [X:%d Y:%d] [P:%d] = %f\n", i, j, (i * N + j), b[(i*N+j)]);
}

void MatrixMultiplication(float* A, float* B, float* C, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

void generateMatrix(float* a, float*b, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            a[i * N + j] = i + j;
            b[i * N + j] = i - j;
        }
}

void showMatrix(float* a, int N){
    for (int i = 0; i < N*N; i++)
        if(i%N == 0){
            printf("\n | %f", a[i]);
        }
        else {
            printf(" | %f", a[i]);
        }
}

void showDiagonalMatrix(float* a, int N) {
    for (int i = 0; i < N; i++) {
        printf(" | %f", a[i+i*N]);
    }
}


int main()
{
    printf("Practica 8");
    printf("\n-----------------------------------------------------------------------------------------------");
    int laps = 1;

    const int N = 1024;
    float* a = new float[N * N];
    float* b = new float[N * N];
    float* c = new float[N * N];

    generateMatrix(a, b, N);

    
    //Add vectors in parallel.
    cudaError_t cudaStatus1 = addWithCudaFloatTimesBlocks(c, a, b, N, laps);
    if (cudaStatus1 != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus1 = cudaDeviceReset();
    if (cudaStatus1 != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    showDiagonalMatrix(c, N);

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCudaDouble(double* c, double* a, double* b, unsigned int size)
{
    double* dev_a = 0;
    double* dev_b = 0;
    double* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernelDoubleShowIndexBlocks <<< 10, 10 >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCudaFloat(float* c, float* a, float* b, unsigned int size)
{
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadsPerBlock(size, size);
    dim3 numBlocks(size / threadsPerBlock.x, size / threadsPerBlock.y);
    multMatrixKernelFloat <<< numBlocks, threadsPerBlock >>> (dev_c, dev_a, dev_b, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCudaFloatTimes(float* c, float* a, float* b, unsigned int size, int laps)
{
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    //Time Recorder
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadsPerBlock(size, size);
    dim3 numBlocks(size / threadsPerBlock.x, size / threadsPerBlock.y);

    cudaEventRecord(start);
    for (int i = 0; i < laps; i++) {
        multMatrixKernelFloat <<< numBlocks, threadsPerBlock >>> (dev_c, dev_a, dev_b, size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("\nTiempo: %f\n\n", (milliseconds/(float)laps)/(float)1000);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCudaFloatTimesBlocks(float* c, float* a, float* b, unsigned int size, int laps)
{
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;
    int threads = 32;
    cudaError_t cudaStatus;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    //Time Recorder
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadsPerBlock(threads, threads, 1);
    dim3 numBlocks(size / threadsPerBlock.x /2, size/ threadsPerBlock.y/2 , 1);

    cudaEventRecord(start);
    for (int i = 0; i < laps; i++) {
        //multMatrixKernelFloatBlocks <<< numBlocks, threadsPerBlock >> > (dev_c, dev_a, dev_b, size);
        for (int e = 0; e < 32 ; e+=16 ) {
            multMatrixKernelFloatBlocks2 <<< numBlocks, threadsPerBlock >>> (dev_c, dev_a, dev_b, size, e);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("\nTiempo: %f\n\n", (milliseconds / (float)laps)/(float)1000);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}



//------------PRACTICA-2------------------------------------------------------------------
/*
int main()
{
    const int length = 100;
    double a1[length];
    double b1[length];
    double c1[length];

    for (int i = 0; i < length; i++) {
        a1[i] = double(i);
        b1[i] = double(i) * (double)2;
    }
    printf("Practica 2");
    printf("\n-----------------------------------------------------------------------------------------------\n");
    // Add vectors in parallel.
    cudaError_t cudaStatus1 = addWithCudaDouble(c1, a1, b1, length);
    if (cudaStatus1 != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus1 = cudaDeviceReset();
    if (cudaStatus1 != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

   
    double escalar = 0;
    for (int i = 0; i < length; i++) {
        escalar += c1[i];
    }
    //printf("Practica 2");
    //printf("\n-----------------------------------------------------------------------------------------------");
    printf("\nEscalar(A1,B1): %1f", escalar);
    printf("\n\n");

    return 0;
}*/



//------------PRACTICA-3------------------------------------------------------------------
/*
    int main()
{
    const int N = 3;
    float* a;
    float* b;
    float* c;

    a = new float[N * N];
    b = new float[N * N];
    c = new float[N * N];

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            a[i * N + j] = i + j;
            b[i * N + j] = i - j;
        }

    MatrixMultiplication(a, b, c, N);
    //showDiagonalMatrix(c, N);
    showMatrix(a, N);
    printf("\n");
    showMatrix(b, N);
    printf("\n");
    showMatrix(c, N);
    printf("\n");

    //printf("Escalar(A1,B1): %1f", escalar);

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
   */


//------------PRACTICA-4------------------------------------------------------------------
/*
int main()
{
    clock_t start, stop;

    const int N = 32;
    float* a;
    float* b;
    float* c;

    a = new float[N * N];
    b = new float[N * N];
    c = new float[N * N];

    generateMatrix(a, b, N);

    start = clock();
    for (int i = 0; i < 1; i++) MatrixMultiplication(a, b, c, N);
    stop = clock();
    printf("Practica 4");
    printf("\n-----------------------------------------------------------------------------------------------\n");
    printf("Tiempo secuencial: %f segundos\n",
        (float)(stop - start) / CLOCKS_PER_SEC / 1);
    getchar();
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
*/

//------------PRACTICA-5------------------------------------------------------------------
/*
int main()
{
    const int N = 3;
    float a[N * N];
    float b[N * N];
    float c[N * N];

    generateMatrix(a, b, N);

    //Add vectors in parallel.
    cudaError_t cudaStatus1 = addWithCudaFloat(c, a, b, N);
    if (cudaStatus1 != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus1 = cudaDeviceReset();
    if (cudaStatus1 != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    showMatrix(c, N);

    return 0;
}
*/

//------------PRACTICA-6------------------------------------------------------------------
/*
int main()
{
    int laps = 5;

    const int N = 32;
    float a[N * N];
    float b[N * N];
    float c[N * N];

    generateMatrix(a, b, N);


    printf("Practica 6");
    printf("\n-----------------------------------------------------------------------------------------------\n");
    //Add vectors in parallel.
    cudaError_t cudaStatus1 = addWithCudaFloatTimes(c, a, b, N, laps);
    if (cudaStatus1 != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus1 = cudaDeviceReset();
    if (cudaStatus1 != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    showMatrix(c, N);
    //getchar();
    return 0;
}
*/

//------------PRACTICA-7------------------------------------------------------------------
/*
int main()
{
    int laps = 10000;

    const int N = 3;
    float a[N * N];
    float b[N * N];
    float c[N * N];

    generateMatrix(a, b, N);

    //Add vectors in parallel.
    cudaError_t cudaStatus1 = addWithCudaFloatTimes(c, a, b, N, laps);
    if (cudaStatus1 != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus1 = cudaDeviceReset();
    if (cudaStatus1 != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    showDiagonalMatrix(c, N);
    //getchar();
    return 0;
}
*/
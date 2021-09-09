
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <mkl.h>

#include <fstream>
#include <sstream>

void load_csv(int n, int m, float* data) {
    std::ifstream file("waveform.csv");

    for (int row = 0; row < n; row++)
    {
        std::string line;
        std::getline(file, line);
        if (!file.good())
            break;

        std::stringstream iss(line);

        for (int col = 0; col < m; col++)
        {
            std::string val;
            std::getline(iss, val, ',');
            if (!iss.good())
                break;

            std::stringstream convertor(val);
            convertor >> data[row * m + col];
        }
    }
}

void showMatrix(int coll, float* C, int m, int e, int n) {
    for (int i = 0; i < n; i++) {
        if (i % coll == 0) {
            printf("\n | %1f", C[i]);
        }
        else {
            printf(" | %1f", C[i]);
        }
    }

    printf("\n           ...            ...            ...            ...            ...            ...            ... \n");
    for (int i = m; i < m+n; i++) {
        if (i % coll == 0) {
            printf("\n | %1f", C[i]);
        }
        else {
            printf(" | %1f", C[i]);
        }
    }

    printf("\n           ...            ...            ...            ...            ...            ...            ... \n");
    for (int i = e-n; i < e; i++) {
        if (i % coll == 0) {
            printf("\n | %1f", C[i]);
        }
        else {
            printf(" | %1f", C[i]);
        }
    }
    printf("\n");
}

cudaError_t addWithCudaFloat(float* c, const float* a, const float* b, const int width, const int P, const int Q);

__global__ void multMatrixKernelFloatBlocks2(float* C, float* A, float* B, int width, int P, int Q) {
    int r = blockDim.x * blockIdx.x + threadIdx.x;
    int c = blockDim.y * blockIdx.y + threadIdx.y;

    if (r < P && c < Q) {
        float value = 0;
        for (int k = 0; k < width; k++) {
            value += A[r * width + k] * B[k * Q + c];
        }
        C[r * Q + c] = value;
    }
}


int main()
{
    /*----- - PARTE 1------------------------------------------------------------
    Cargar los datos
    */
    const int nRow = 5000;
    const int nColl = 10;
    float* datas = new float[nRow * nColl];
    load_csv(nRow, nColl, datas);

    /*----- - PARTE 2------------------------------------------------------------
    Centrar los datos restando la media de cada componente, generando una matriz XC
    */
    const MKL_INT incx = 0;
    float media;
    const float a = 1;

    for (int i = 0; i < nColl; i++) {
        media = -(cblas_sdot(nRow, &datas[i], nColl, &a, incx) / (float)nRow); // Sacar media
        cblas_saxpy(nRow, a, &media, incx, &datas[i], nColl); // generando una matriz XC
    }

    /*------ - PARTE 4------------------------------------------------------------
        Copiamos manualmente los autovectores por un problema de configuración que
        no pudimos solucionar, pero se generan por funciones de blas en la otra parte
        del proyecto
    */
    float AV[20] = { -0.0008, 0.0188,
                     0.0933, -0.0802,
                     0.1757, -0.1562,
                     0.2676, -0.2553,
                     0.3695, -0.3317,
                     0.4412, -0.1902,
                     0.5211, -0.0628,
                     0.4131, 0.2222,
                     0.3108, 0.5137,
                     0.1373, 0.6635
    };

    float AVT[100] = { -0.0008,    0.0188,    0.3372,    0.5261,   -0.1617,    0.0808,    0.7478,    0.0105,   -0.1225,    0.0471,
                        0.0933,   -0.0802,   -0.0393,    0.3523,   -0.4942,   -0.3229,   -0.3664,    0.0849,   -0.1738,    0.5823,
                        0.1757,   -0.1562,  -0.1378,    -0.1354,   -0.2342,    0.7978,    0.0233,    0.3597,    0.1246,    0.2659,
                        0.2676,   -0.2553,    0.5977,   -0.4469,    0.1129,   -0.1616,    0.0190,    0.3228,   -0.3941,    0.0861,
                        0.3695,   -0.3317,   -0.1881,    0.2403,    0.3342,   -0.3163,    0.1062,    0.4373,    0.4960,   -0.0038,
                        0.4412,   -0.1902,    0.3386,    0.3923,   -0.0193,    0.2641,   -0.4235,   -0.3052,    0.0205,   -0.3971,
                        0.5211,   -0.0628,   -0.1344,   -0.2972,   -0.0063,   -0.0619,    0.2846,   -0.6302,    0.1248,    0.3471,
                        0.4131,    0.2222,  -0.5104,    0.1172,    0.1228,     0.0122,     0.0722,    0.1642,    -0.6463,    -0.2014,
                        0.3108,    0.5137,    0.1071,   -0.2102,   -0.5517,   -0.1856,    0.0550,    0.2141,    0.3049,   -0.3213,
                        0.1373,    0.6635,    0.2616,    0.1518,   0.4801,     0.1356,    -0.1608,    0.0695,    0.1061,    0.3982 
    };

    const int width = 10;
    const int P = 5000;
    const int Q = 2;

    float* C = new float[P * Q];

    //Add vectors in parallel.
    cudaError_t cudaStatus = addWithCudaFloat(C, datas, AV, width, P, Q);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    showMatrix(2, C, 5000, 10000, 10);

    const int Q2 = 10;
    float* datas2 = new float[nRow * nColl];
    float* C2 = new float[P * Q2];

    load_csv(nRow, nColl, datas2);

    //Add vectors in parallel.
    cudaError_t cudaStatus1 = addWithCudaFloat(C2, datas2, AVT, width, P, Q2);
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

    showMatrix(10, C2, 25000, 50000, 50);

    delete[]C;
    delete[]C2;
    delete[]datas;
    delete[]datas2;

    return 0;
}

cudaError_t addWithCudaFloat(float* c, const float* a, const float* b, const int width, const int P, const int Q)
{
    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;
    const int sizeC = P * Q;
    const int sizeA = P * width;
    const int sizeB = width * Q;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, sizeC * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, sizeA * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, sizeB * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, sizeA * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, sizeB * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadsPerBlock(32,32, 1); //Ponemos el maximo numero de hilos por bloque
    dim3 numBlocks(157, 1);         //Ya que tenemos que en las dos multiplicaciones de matrices 
                                    //generaremos una de 5000*10 y otra de 5000*2 no necesitamos
                                    //mas de 157*1 bloques
    multMatrixKernelFloatBlocks2 << <numBlocks, threadsPerBlock >> > (dev_c, dev_a, dev_b, width, P, Q);
    
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
    cudaStatus = cudaMemcpy(c, dev_c, sizeC * sizeof(float), cudaMemcpyDeviceToHost);
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
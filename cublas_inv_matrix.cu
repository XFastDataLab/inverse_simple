
#include "def.h"
#include "cublas_v2.h"

int my_gauss_inverse_gpu_by_cublas(float** d_in, int size, float** d_out, int my_np) {

	
	cudaError_t cudaerror = cudaSuccess;

	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	int* info, * pivo;
	
	cudaMalloc((void**)&info, sizeof(int) * my_np);
	cudaMalloc((void**)&pivo, sizeof(int) * size * my_np);


	float** gpuMat;
	cudaMalloc((void**)&gpuMat, sizeof(float*) * my_np);
	cudaMemcpy(gpuMat, d_in, sizeof(float*) * my_np, cudaMemcpyHostToDevice);

	
	float** gpuInvMat;
	cudaMalloc((void**)&gpuInvMat, sizeof(float*) * my_np);
	cudaMemcpy(gpuInvMat, d_out, sizeof(float*) * my_np, cudaMemcpyHostToDevice);

	
	cublasSgetrfBatched(handle, size, gpuMat, size, pivo, info, my_np);

	cudaDeviceSynchronize();
	cublasSgetriBatched(handle, size, gpuMat, size, pivo, gpuInvMat, size, info, my_np);

	cudaDeviceSynchronize();
	
	cudaFree(info);
	cudaFree(pivo);
	cudaFree(gpuMat);
	cudaFree(gpuInvMat);
	/*for (int i = 0; i < my_np; i++) {
		cudaFree(resulthd[i]);
	}*/

	return 1;
}

#include "def.h"
#include "cublas_v2.h"

int my_gauss_inverse_gpu_by_cublas(double** d_in, int size, double** d_out, int my_np) {

	cudaEvent_t start, stop;
	float time_elapsed = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaError_t cudaerror = cudaSuccess;

	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	int* info, * pivo;
	
	cudaMalloc((void**)&info, sizeof(int) * my_np);
	cudaMalloc((void**)&pivo, sizeof(int) * size * my_np);

	double** srchd = new double* [my_np];
	for (int i = 0; i < my_np; i++) {
		cudaMalloc((void**)&srchd[i], sizeof(double) * size * size);
		cudaMemcpy(srchd[i], d_in[i], sizeof(double) * size * size, cudaMemcpyHostToDevice);
	}

	double** gpuMat;
	cudaMalloc((void**)&gpuMat, sizeof(double*) * my_np);
	cudaMemcpy(gpuMat, srchd, sizeof(double*) * my_np, cudaMemcpyHostToDevice);


	double** resulthd = new double* [my_np];
	for (int i = 0; i < my_np; i++) {
		cudaMalloc((void**)&resulthd[i], sizeof(double) * size * size);
	}

	
	double** gpuInvMat;
	cudaMalloc((void**)&gpuInvMat, sizeof(double*) * my_np);
	cudaMemcpy(gpuInvMat, resulthd, sizeof(double*) * my_np, cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);
	cublasDgetrfBatched(handle, size, gpuMat, size, pivo, info, my_np);

	cudaDeviceSynchronize();
	cublasDgetriBatched(handle, size, gpuMat, size, pivo, gpuInvMat, size, info, my_np);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	
	for (int i = 0; i < my_np; i++) {
		cudaMemcpy(d_out[i], resulthd[i], sizeof(double) * size * size, cudaMemcpyDeviceToHost);
	}

	

	cudaEventSynchronize(start);    //Waits for an event to complete.
	cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
	cudaEventElapsedTime(&time_elapsed, start, stop);    //计算时间差
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("执行时间：%f(ms)\n", time_elapsed);

	writeGPUResults(time_elapsed);

	cudaFree(info);
	cudaFree(pivo);
	cudaFree(gpuMat);
	cudaFree(gpuInvMat);
	for (int i = 0; i < my_np; i++) {
		cudaFree(srchd[i]);
		cudaFree(resulthd[i]);
	}

	return 1;
}
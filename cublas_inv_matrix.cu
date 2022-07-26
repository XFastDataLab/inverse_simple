
#include "def.h"
#include "cublas_v2.h"

int my_gauss_inverse_gpu_by_cublas(float** d_in, int size, float** d_out, int my_np) {

	
	cudaError_t cudaerror = cudaSuccess;

	cublasHandle_t handle;
	cudaEvent_t start, stop;
	float time_elapsed = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	tools_gpuAssert(cudaEventRecord(start, 0));

	cublasCreate_v2(&handle);
	int* info, * pivo;
	
	cudaMalloc((void**)&info, sizeof(int) * my_np);
	cudaMalloc((void**)&pivo, sizeof(int) * size * my_np);


	float** gpuMat;
	cudaMalloc((void**)&gpuMat, sizeof(float*) * my_np);
	cudaMemcpy(gpuMat, d_in, sizeof(float*) * my_np, cudaMemcpyHostToDevice);

	float** resulthd = new float* [my_np];
	for (int i = 0; i < my_np; i++) {
		cudaMalloc((void**)&resulthd[i], sizeof(float) * size * size);
	}
	
	float** gpuInvMat;
	cudaMalloc((void**)&gpuInvMat, sizeof(float*) * my_np);
	cudaMemcpy(gpuInvMat, resulthd, sizeof(float*) * my_np, cudaMemcpyHostToDevice);

	
	cublasSgetrfBatched(handle, size, gpuMat, size, pivo, info, my_np);

	cudaDeviceSynchronize();
	cublasSgetriBatched(handle, size, gpuMat, size, pivo, gpuInvMat, size, info, my_np);

	cudaDeviceSynchronize();

	tools_gpuAssert(cudaEventRecord(stop, 0));
	cudaEventSynchronize(start);    //Waits for an event to complete.
	cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
	tools_gpuAssert(cudaEventElapsedTime(&time_elapsed, start, stop));    //计算时间差
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("执行时间：%f(ms)\n", time_elapsed);

	for (int i = 0; i < my_np; i++) {
		cudaMemcpy(d_out[i], resulthd[i], sizeof(float) * size * size, cudaMemcpyDeviceToHost);
	}
	
	writeGPUResults(time_elapsed);

	cudaFree(info);
	cudaFree(pivo);
	cudaFree(gpuMat);
	cudaFree(gpuInvMat);

	for (int i = 0; i < my_np; i++) {
		cudaFree(resulthd[i]);
	}

	cublasDestroy(handle);

	return 1;
}
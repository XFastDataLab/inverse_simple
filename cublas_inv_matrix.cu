
#include "def.h"
#include "cublas_v2.h"

int my_gauss_inverse_gpu_by_cublas(float** d_in, int size, float** d_out, int my_np) {

	
	cudaError_t cudaerror = cudaSuccess;

	cublasHandle_t handle;
	cudaEvent_t start, stop, start1,end1, start2,end2;
	float time_elapsed = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&start1);
	cudaEventCreate(&end1);
	cudaEventCreate(&start2);
	cudaEventCreate(&end2);


	

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

	tools_gpuAssert(cudaEventRecord(start, 0));
	cublasSgetrfBatched(handle, size, gpuMat, size, pivo, info, my_np);
	cudaThreadSynchronize();
	cudaDeviceSynchronize();
	cublasSgetriBatched(handle, size, gpuMat, size, pivo, gpuInvMat, size, info, my_np);
	cudaThreadSynchronize();
	cudaDeviceSynchronize();

	tools_gpuAssert(cudaEventRecord(stop, 0));


	cudaEventSynchronize(start);    //Waits for an event to complete.
	cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
	tools_gpuAssert(cudaEventElapsedTime(&time_elapsed, start, stop));    //计算时间差



	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaEventDestroy(start1);
	cudaEventDestroy(end1);
	cudaEventDestroy(start2);
	cudaEventDestroy(end2);


	printf("执行时间：%f(ms)\n", time_elapsed);
	cout << "before" << endl;
	for (int i = 0; i < my_np; i++) {
		cudaMemcpy(d_out[i], resulthd[i], sizeof(float) * size * size, cudaMemcpyDeviceToHost);
	}
	cout << "memcpy" << endl;
	writeGPUResults(time_elapsed);

	cudaFree(info);
	cudaFree(pivo);
	cudaFree(gpuMat);
	cudaFree(gpuInvMat);

	for (int i = 0; i < my_np; i++) {
		cudaFree(resulthd[i]);
	}

	cublasDestroy(handle);
	cout << "hhh" << endl;
	return 1;
}
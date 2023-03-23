
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
	
	tools_gpuAssert(cudaMalloc((void**)&info, sizeof(int) * my_np));
	tools_gpuAssert(cudaMalloc((void**)&pivo, sizeof(int) * size * my_np));


	float*gpuMat, **A = (float**)malloc(my_np * sizeof(float*)), **A_d;
	tools_gpuAssert(cudaMalloc((void**)&gpuMat, sizeof(float) * my_np * size * size));
	tools_gpuAssert(cudaMalloc((void**)&A_d, sizeof(float*) * my_np));

	A[0] = gpuMat;
	for (int i = 1; i < my_np; i++) {
		A[i] = A[i - 1] + size * size;
	}

	tools_gpuAssert(cudaMemcpy(A_d, A, my_np * sizeof(float*), cudaMemcpyHostToDevice));
	
	for (int i = 0; i < my_np; i++) {
		tools_gpuAssert(cudaMemcpy(gpuMat + (i * size * size), d_in[i], size * size * sizeof(float), cudaMemcpyHostToDevice));
	}


	float** C = (float**)malloc(my_np * sizeof(float*));
	float** C_d, * C_dflat;

	cudaMalloc(&C_d, my_np * sizeof(float*));
	cudaMalloc(&C_dflat, size * size * my_np * sizeof(float));
	C[0] = C_dflat;
	for (int i = 1; i < my_np; i++) {
		C[i] = C[i - 1] + (size * size);
	}

	tools_gpuAssert(cudaMemcpy(C_d, C, my_np * sizeof(float*), cudaMemcpyHostToDevice));



	tools_gpuAssert(cudaEventRecord(start, 0));
	cublasSgetrfBatched(handle, size, A_d, size, pivo, info, my_np);
	cudaThreadSynchronize();
	cudaDeviceSynchronize();
	cublasSgetriBatched(handle, size, A_d, size, pivo, C_d, size, info, my_np);
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
		cudaMemcpy(d_out[i], C_dflat+(i*size*size), sizeof(float) * size * size, cudaMemcpyDeviceToHost);
	}
	cout << "memcpy" << endl;
	writeGPUResults(time_elapsed);

	cudaFree(info);
	cudaFree(pivo);
	cudaFree(gpuMat);


	free(A);
	cudaFree(A_d);
	cudaFree(C_d);
	cudaFree(C_dflat);
	free(C);
	cublasDestroy(handle);
	cout << "hhh" << endl;
	return 1;
}
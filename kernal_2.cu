
#include "def.h"

static __global__
void kernal_function_2(__DATA_TYPE* out, int size) {

	int idx = threadIdx.x;
	int idy = threadIdx.y;
	int bx = blockIdx.x;

	__shared__ __DATA_TYPE A[2][2];
	A[idx][idy] = out[bx * size * size + idx * size + idy];
	__syncthreads();
	////out[bx * size * size] = 1 / A[0][0] + ((A[0][1] * 1 / A[0][0] * 1 / (A[1][1] + A[1][0] * 1 / A[0][0] * (-1) * A[0][1])) * (-1)) * (A[1][0] * 1 / A[0][0] * (-1));
	out[bx * size * size] = 1 / (A[1][1] + A[1][0] * 1 / A[0][0] * (-1) * A[0][1]);
	out[bx * size * size + 1]=(A[0][1] * 1 / A[0][0] * 1 / (A[1][1] + A[1][0] * 1 / A[0][0] * (-1) * A[0][1]))* (-1);
	out[bx * size * size + 2] = A[1][0] * 1 / A[0][0] * (-1) * 1 / (A[1][1] + A[1][0] * 1 / A[0][0] * (-1) * A[0][1]);
	out[bx * size * size + 3] = 1 / (A[1][1] + A[1][0] * 1 / A[0][0] * (-1) * A[0][1]);


}


int inverse_gauss_func_2(__DATA_TYPE* out, int size, int my_np) {

	int deviceSno = 0;
	cudaSetDevice(deviceSno);

	cudaEvent_t start, stop;
	cudaError_t cudaerror = cudaSuccess;

	DeviceInfo d;
	GetDeviceInfo(d);


	__DATA_TYPE* d_out;

	//cudaEventRecord(start, 0);
	gpuErrchk(cudaMalloc((void**)&d_out, my_np * size * size * sizeof(__DATA_TYPE)));
	gpuErrchk(cudaMemcpy(d_out, out, my_np * size * size * sizeof(__DATA_TYPE), cudaMemcpyHostToDevice));

	dim3 blocks(my_np), threads(size,2);
	

	float time_elapsed = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start, 0);
	kernal_function_2 << <blocks, threads >> > (d_out, size);
	cudaEventRecord(stop, 0);

	gpuErrchk(cudaMemcpy(out, d_out, my_np * size * size * sizeof(__DATA_TYPE), cudaMemcpyDeviceToHost));
	cudaEventSynchronize(start);    //Waits for an event to complete.
	cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务

	cudaEventElapsedTime(&time_elapsed, start, stop);    //计算时间差
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_out);

	printf("执行时间：%f(ms)\n", time_elapsed);
	cudaerror = cudaGetLastError();

	free_device_list(d.device);
	//record time
	writeGPUResults(time_elapsed);

	if (cudaerror != cudaSuccess) {
		cudaCheck(cudaerror);
		return 0;
	}
	return 1;

}
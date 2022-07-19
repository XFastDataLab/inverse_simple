/*
 *
 * Copyright (C) 2020-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

#include "def.h"


//static __device__
//void Matrix_copy_glob2shr(__DATA_TYPE* glob_data, __DATA_TYPE* shr_data, int size) {
//
//	for (int i = 0; i < size; i++) {
//		shr_data[i] = glob_data[i];
//	}
//}
//
//static __device__
//void Matrix_copy_shr2glob(__DATA_TYPE* shr_data, __DATA_TYPE* glob_data, int size) {
//
//	for (int i = 0; i < size; i++) {
//		glob_data[i] = shr_data[i];
//	}
//}

 /*
 *
 * Gauss求逆，使用O(n^3)的时间复杂度，尽量让所有线程都做同样的工作
 **/
static __global__
void Gauss_Jordan_Inverse(__DATA_TYPE* mat_tmp, int size, int dy) {

	int idx = threadIdx.x;
	int idy = threadIdx.y;
	int bx = blockIdx.x;


	if (idx >= size || idy >= dy) return;

	mat_tmp += bx * size * size * dy;
	extern __shared__ __DATA_TYPE out[]; //using static shared memory 48 KB
	Matrix_copy_glob2shr(mat_tmp, out, size*size);
	__syncthreads();

	int i, j, k;
	__DATA_TYPE c;
	for (k = 0; k < size; k++) {
		//1.m(k,k) = 1/m(k,k)
		out[k * size + k] = 1.0 / out[k * size + k];
		c = out[k * size + k];
		//2.m(i,k) = -m(k,k) * m(i,k), i!=k
		for (i = 0; i < k; i++)    out[i * size + k] *= -1 * c;
		for (i = k + 1; i < size; i++)    out[i * size + k] *= -1 * c;

		//3.m(i,j) = m(i,j) + m(i,k) * m(k,j), i,j != k
		for (i = 0; i < k; i++) {
			for (j = 0; j < k; j++)   out[i * size + j] += out[i * size + k] * out[k * size + j];
			for (j = k + 1; j < size; j++)   out[i * size + j] += out[i * size + k] * out[k * size + j];
		}
		for (i = k + 1; i < size; i++) {
			for (j = 0; j < k; j++)   out[i * size + j] += out[i * size + k] * out[k * size + j];
			for (j = k + 1; j < size; j++)   out[i * size + j] += out[i * size + k] * out[k * size + j];
		}

		//4.m(k,j) = m(k,k) * m(k,j), j != k
		for (j = 0; j < k; j++)    out[k * size + j] *= c;
		for (j = k + 1; j < size; j++)    out[k * size + j] *= c;

	}

	Matrix_copy_shr2glob(out, mat_tmp, size*size);
}


static
int single_sm_inverse_gauss_gpu(__DATA_TYPE* out, int size, int my_np) {

	int deviceSno = 0;
	cudaSetDevice(deviceSno);

	cudaEvent_t start, stop;
	cudaError_t cudaerror = cudaSuccess;

	DeviceInfo d;
	GetDeviceInfo(d);
	Device* ds = d.device;
	while (ds) {
		if (ds->index == deviceSno)
			break;
		else ds = ds->next;
	}
	int maxbytes = ds->sharedMemPerBlockOptin; // 65535 byte = 64 KB
	cudaFuncSetAttribute(Gauss_Jordan_Inverse, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
	int dy = 1; //dy->[1,1024]

	//Output some necessary infomation for remind you!
	float elapse_time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	__DATA_TYPE* d_out;

	//cudaEventRecord(start, 0);
	gpuErrchk(cudaMalloc((void**)&d_out, my_np * size * size * sizeof(__DATA_TYPE)));
	gpuErrchk(cudaMemcpy(d_out, out, my_np * size * size * sizeof(__DATA_TYPE), cudaMemcpyHostToDevice));

	dim3 blocks(my_np / dy), threads(dy);
	cudaEventRecord(start, 0);
	Gauss_Jordan_Inverse << <blocks, threads, maxbytes >> > (d_out, size, dy);
	cudaEventRecord(stop, 0);

	gpuErrchk(cudaMemcpy(out, d_out, my_np * size * size * sizeof(__DATA_TYPE), cudaMemcpyDeviceToHost));
	cudaEventSynchronize(start);    //Waits for an event to complete.
	cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务

	cudaEventElapsedTime(&elapse_time, start, stop);    //计算时间差
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_out);

	printf("执行时间：%f(ms)\n", elapse_time);
	cudaerror = cudaGetLastError();

	free_device_list(d.device);
	//record time
	writeGPUResults(elapse_time);

	if (cudaerror != cudaSuccess) {
		cudaCheck(cudaerror);
		return 0;
	}
	return 1;
}

static
int more_sm_inverse_gauss_gpu(__DATA_TYPE* out, int size, int my_np) {

	int deviceSno = 0;
	cudaSetDevice(deviceSno);

	cudaEvent_t start, stop;
	cudaError_t cudaerror = cudaSuccess;
	float elapse_time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);



	DeviceInfo d;
	GetDeviceInfo(d);
	Device* ds = d.device;
	while (ds) {
		if (ds->index == deviceSno)
			break;
		else ds = ds->next;
	}
	int maxbytes = ds->sharedMemPerBlockOptin; // 65535 byte = 64 KB
	cudaFuncSetAttribute(Gauss_Jordan_Inverse, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);

	int sm = ds->SMCount;
	int tasksPerStream = my_np / sm;
	int dis = tasksPerStream * size * size;


	int dy =1; //dy->[1,1024]


	__DATA_TYPE* d_out;

	cudaStream_t* stream = new cudaStream_t[sm];
	for (int i = 0; i < sm; i++) {
		gpuErrchk(cudaStreamCreate(&stream[i]));
	}


	dim3 blocks(tasksPerStream / dy), threads(dy);


	gpuErrchk(cudaMalloc((void**)&d_out, my_np * size * size * sizeof(__DATA_TYPE)));

	for (int i = 0; i < sm; i++) {
		gpuErrchk(cudaMemcpyAsync(d_out + i * dis, out + i * dis, dis * sizeof(__DATA_TYPE), cudaMemcpyHostToDevice, stream[i]));
	}

	cudaEventRecord(start, 0);
	for (int i = 0; i < sm; i++) {
		Gauss_Jordan_Inverse << <blocks, threads, maxbytes, stream[i] >> > (d_out + i * dis, size, dy);
	}
	cudaEventRecord(stop, 0);

	for (int i = 0; i < sm; i++) {
		gpuErrchk(cudaMemcpyAsync(out + i * dis, d_out + i * dis, dis * sizeof(__DATA_TYPE), cudaMemcpyDeviceToHost, stream[i]));
	}

	cudaerror = cudaGetLastError();

	for (int i = 0; i < sm; ++i) {
		gpuErrchk(cudaStreamDestroy(stream[i]));
	}

	free_device_list(d.device);
	gpuErrchk(cudaFree(d_out));


	cudaEventSynchronize(start);    //Waits for an event to complete.
	cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
	cudaEventElapsedTime(&elapse_time, start, stop);    //计算时间差
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	writeGPUResults(elapse_time);

	printf("执行时间：%f(ms)\n", elapse_time);
	if (cudaerror != cudaSuccess) {
		return 0;
	}
	return 1;


}

static
int get_devided_number_of_single_to_mul_sm(int size, int my_np) {
	int deviceSno = 0;
	DeviceInfo d;
	GetDeviceInfo(d);
	Device* ds = d.device;
	while (ds) {
		if (ds->index == deviceSno)
			break;
		else ds = ds->next;
	}
	int maxbytes = ds->sharedMemPerBlockOptin; // 65535 byte = 64 KB
	int dy = min(floor(ds->sharedMemPerBlockOptin * 1.0 / (size * size * sizeof(__DATA_TYPE))), my_np);
	dy = min(1024 / size, dy); //dy->[1,1024]
	free_device_list(d.device);
	return ceil(dy * 65536 * 1.0 / 35);
}


int my_single_block_single_gauss_inverse_gpu(__DATA_TYPE* out, int size, int my_np) {

	if (size > 90) {
		printf("ERROR!!! The method allow the size of matrix small than 90!!!\n");
		return 0;
	}

	if (my_np >= 1 && my_np < 1024) {
		return single_sm_inverse_gauss_gpu(out, size, my_np);
	}
	else {
		return more_sm_inverse_gauss_gpu(out, size, my_np);
	}
}
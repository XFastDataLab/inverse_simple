/*
 *
 * Copyright (C) 2020-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

#include "def.h"

__device__ __host__
void printMatrix_ct(__DATA_TYPE* data, int size, int iter) {
	printf("µÚ%d´ÎÑ­»·\n", iter);
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			printf("%lf ", data[i * size + j]);
		}
		printf("\n");
	}
}

__device__
void Matrix_copy_glob2shr(__DATA_TYPE* glob_data, __DATA_TYPE* shr_data, int size) {

	for (int i = 0; i < size; i++) {
		shr_data[i] = glob_data[i];
	}
}

__device__
void Matrix_copy_shr2glob(__DATA_TYPE* shr_data, __DATA_TYPE* glob_data, int size) {

	for (int i = 0; i < size; i++) {
		glob_data[i] = shr_data[i];
	}
}


/// <summary>
/// resolve remainder cannot be computed.
/// </summary>
/// <param name="index">[0...n-1]</param>
/// <param name="n"></param>
/// <param name="sum"></param>
/// <returns></returns>
int getAmountOfTasks(int index, int n, int sum) {
	int span = sum / n;
	int remain = sum % n;
	if (index < remain) return span + 1;
	else return span;
}





void cudaMemcpyToDeviceControl(__DATA_TYPE* d_out, __DATA_TYPE* out, size_t count,int my_n,int my_np) {
	static int i = getConfigInt(USE_COPY_MATRIX_C);
	if (i == 0) { // not use copied matrix.
		gpuErrchk(cudaMemcpy(d_out, out, count, cudaMemcpyHostToDevice));
	}
	else { // use copied matrix.
		static int size = my_n * my_n;
		for (int j = 0; j < my_np; j++) {
			gpuErrchk(cudaMemcpy(d_out + j * size, out, size * sizeof(__DATA_TYPE), cudaMemcpyHostToDevice));
		}
	}
}


void cudaMemcpyToHostControl(__DATA_TYPE* d_out, __DATA_TYPE* out, size_t count,int my_n) {
	static int i = getConfigInt(USE_COPY_MATRIX_C);
	if (i == 0) {
		gpuErrchk(cudaMemcpy(out, d_out, count, cudaMemcpyDeviceToHost));
	}
	else {
		static int size = my_n * my_n;
		gpuErrchk(cudaMemcpy(out, d_out, size * sizeof(__DATA_TYPE), cudaMemcpyDeviceToHost));
	}
}


void cudaMemcpyToDeviceMulStreamsControl(__DATA_TYPE* d_out, __DATA_TYPE* out, int tasksPerStream, int sm, cudaStream_t* stream,int my_n) {
	static int i = getConfigInt(USE_COPY_MATRIX_C);
	int area = my_n * my_n;
	int dis = tasksPerStream * area;
	__DATA_TYPE* out_copy;

	if (i == 0) {
		for (int i = 0; i < sm; i++) {
			gpuErrchk(cudaMemcpyAsync(d_out + i * dis, out + i * dis, dis * sizeof(__DATA_TYPE), cudaMemcpyHostToDevice, stream[i]));
		}
	}
	else {

		out_copy = (__DATA_TYPE*)malloc(sizeof(__DATA_TYPE) * dis);
		for (int k = 0; k < tasksPerStream; ++k) {
			memcpy(out_copy + k * area, out, area * sizeof(__DATA_TYPE));
		}


		for (int i = 0; i < sm; i++) {
			gpuErrchk(cudaMemcpyAsync(d_out + i * dis, out_copy, dis * sizeof(__DATA_TYPE), cudaMemcpyHostToDevice, stream[i]));
		}
		free(out_copy);
	}
}


void cudaMemcpyToHostMulStreamsControl(__DATA_TYPE* d_out, __DATA_TYPE* out, int tasksPerStream, int sm, cudaStream_t* stream,int my_n) {
	static int i = getConfigInt(USE_COPY_MATRIX_C);
	int area = my_n * my_n;
	int dis = tasksPerStream * area;

	if (i == 0) {
		for (int i = 0; i < sm; ++i) {
			gpuErrchk(cudaMemcpyAsync(out + i * dis, d_out + i * dis, dis * sizeof(__DATA_TYPE), cudaMemcpyDeviceToHost, stream[i]));
		}
	}
	else {

		//cudaDeviceSynchronize();
		//gpuErrchk(cudaMemcpy(out, d_out + 6 * dis, size * sizeof(__DATA_TYPE), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpyAsync(out, d_out + 3 * dis + 5 * area, area * sizeof(__DATA_TYPE), cudaMemcpyDeviceToHost, stream[3]));
	}
}
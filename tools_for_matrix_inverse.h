/*
 *
 * Copyright (C) 2020-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

#ifndef TOOLS_FOR_MATRIX_INVERSE_H
#define TOOLS_FOR_MATRIX_INVERSE_H


extern __device__ __host__
void printMatrix_ct(__DATA_TYPE* data, int size, int iter);

extern __device__
void Matrix_copy_glob2shr(__DATA_TYPE* glob_data, __DATA_TYPE* shr_data, int size);

extern __device__
void Matrix_copy_shr2glob(__DATA_TYPE* shr_data, __DATA_TYPE* glob_data, int size);

int getAmountOfTasks(int index, int n, int sum);

void cudaMemcpyToDeviceControl(__DATA_TYPE* d_out, __DATA_TYPE* out, size_t count, int my_n, int my_np);

void cudaMemcpyToHostControl(__DATA_TYPE* d_out, __DATA_TYPE* out, size_t count, int my_n);

void cudaMemcpyToDeviceMulStreamsControl(__DATA_TYPE* d_out, __DATA_TYPE* out, int tasksPerStream, int sm, cudaStream_t* stream, int my_n);

void cudaMemcpyToHostMulStreamsControl(__DATA_TYPE* d_out, __DATA_TYPE* out, int tasksPerStream, int sm, cudaStream_t* stream, int my_n);

#endif // !TOOLS_FOR_MATRIX_INVERSE_H

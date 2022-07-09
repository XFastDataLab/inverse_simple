
/*
 *
 * Copyright (C) 2020-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

#include "def.h"


 /*
 *
 * Gauss求逆，使用O(n^2)的时间复杂度，尽量让所有线程都做同样的工作
 **/
static __global__
void Gauss_Jordan_Inverse(__DATA_TYPE* mat_tmp, int size, int dy) {

	int idx = threadIdx.x;
	int idy = threadIdx.y;
	int bx = blockIdx.x;

	int dis = idy * size * size; //Compute the distance between idy-th matrix and the address of mat_tmp.

	if (idx >= size || idy >= dy) return;

	mat_tmp += bx * size * size * dy;
	extern __shared__ __DATA_TYPE mat[]; //using static shared memory 48 KB
	Matrix_copy_glob2shr(mat_tmp + idx * size + dis, mat + idx * size + dis, size);
	__syncthreads();

	int i, j, k;
	__DATA_TYPE c;
	for (k = 0; k < size; k++) {
		//1.m(k,k) = 1/m(k,k)
		mat[k * size + k + dis] = 1.0 / mat[k * size + k + dis];
		c = mat[k * size + k + dis];

		//2.m(i,k) = -m(k,k) * m(i,k), i!=k
		if (idx != k) mat[idx * size + k + dis] *= -1 * c;

		__syncthreads();
		//3.m(i,j) = m(i,j) + m(i,k) * m(k,j), i,j != k
		for (i = 0; i < k; i++) {
			if (idx != k) mat[i * size + idx + dis] += mat[i * size + k + dis] * mat[k * size + idx + dis];
		}
		for (i = k + 1; i < size; i++) {
			if (idx != k) mat[i * size + idx + dis] += mat[i * size + k + dis] * mat[k * size + idx + dis];
		}

		//4.m(k,j) = m(k,k) * m(k,j), j != k
		if (idx != k)  mat[k * size + idx + dis] *= c;
		__syncthreads();
	}

	Matrix_copy_shr2glob(mat + idx * size + dis, mat_tmp + idx * size + dis, size);
}


//static __global__
//void Gauss_Jordan_Inverse(__DATA_TYPE* mat_tmp, int size, int dy) {
//
//	register int idx = threadIdx.x%size;
//	register int idy = threadIdx.x/size;
//	register int bx = blockIdx.x;
//
//	register int dis = idy * size * size; //Compute the distance between idy-th matrix and the address of mat_tmp.
//
//	if (idx >= size || idy >= dy) return;
//
//	mat_tmp += bx * size * size * dy;
//	extern __shared__ __DATA_TYPE mat[]; //using static shared memory 48 KB
//	Matrix_copy_glob2shr(mat_tmp + idx * size + dis, mat + idx * size + dis, size);
//	__syncthreads();
//
//	int i, j, k;
//	__DATA_TYPE c;
//	register int ksize = dis;
//	for (k = 0; k < size; k++) {
//		//1.m(k,k) = 1/m(k,k)
//		register int mid = ksize + k;
//		mat[mid] = 1.0 / mat[mid];
//		c = mat[mid];
//
//		//2.m(i,k) = -m(k,k) * m(i,k), i!=k
//		if (idx != k) mat[idx * size + k + dis] *= -1 * c;
//
//		__syncthreads();
//		//3.m(i,j) = m(i,j) + m(i,k) * m(k,j), i,j != k
//		register int tem = dis;
//		for (i = 0; i < size; i++) {
//			if (idx != k ||i!=k) mat[tem + idx] += mat[tem + k] * mat[ksize + idx];
//			tem += size;
//		}
//
//		//4.m(k,j) = m(k,k) * m(k,j), j != k
//		if (idx != k)  mat[ksize + idx] *= c;
//		__syncthreads();
//		ksize += size;
//	}
//
//	Matrix_copy_shr2glob(mat + idx * size + dis, mat_tmp + idx * size + dis, size);
//}

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
	int dy = min(floor(ds->sharedMemPerBlockOptin * 1.0 / (size * size * sizeof(__DATA_TYPE))), my_np);
	dy = min(1024 / size, dy); //dy->[1,1024]
	int matMaxSize = floor(sqrt(maxbytes * 1.0 / sizeof(__DATA_TYPE)));

	//Output some necessary infomation for remind you!
	int remain = my_np % dy;
	printf("%d matrix inverse works per block,MatMaxSize:%d\n", dy, matMaxSize);
	if (remain) {
		printf("Notice:There have %d last matrixes will not be inversed,You would better set my_np to multiples of %d!!\n", remain, dy);
	}

	float elapse_time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	__DATA_TYPE* d_out;

	//cudaEventRecord(start, 0);
	gpuErrchk(cudaMalloc((void**)&d_out, my_np * size * size * sizeof(__DATA_TYPE)));
	gpuErrchk(cudaMemcpy(d_out, out, my_np * size * size * sizeof(__DATA_TYPE), cudaMemcpyHostToDevice));

	dim3 blocks(my_np / dy), threads(size, dy);
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
	//Compute number of works every block.
	/*int remain = my_np % sm;
	if (remain) {
		printf("MSM Notice:there have %d last matrix(es) won't be inversed,becasuse be ignored. You would be better set my_np to mutilples of %d. \n\n",remain, sm);
	}*/

	int dy = min(floor(ds->sharedMemPerBlockOptin * 1.0 / (size * size * sizeof(__DATA_TYPE))), tasksPerStream);
	dy = min(1024 / size, dy); //dy->[1,1024]
	//int matMaxSize = floor(sqrt(maxbytes * 1.0 / sizeof(__DATA_TYPE)));


	__DATA_TYPE* d_out;

	cudaStream_t* stream = new cudaStream_t[sm];
	for (int i = 0; i < sm; i++) {
		gpuErrchk(cudaStreamCreate(&stream[i]));
	}

	/*printf("%d matrix inverse works per block,MatMaxSize:%d\n", dy, matMaxSize);
	remain = tasksPerStream % dy;
	printf("remain:%d\n", remain);
	if (remain) {
		printf("MulSm Notice:There have %d last matrixes will not be inversed,it will happen every %d matrixes,You would better set my_np to multiples of %d!!\n", remain,tasksPerStream, sm*dy);
	}*/

	dim3 blocks(tasksPerStream / dy), threads(size,dy);


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


int my_algorithm2(__DATA_TYPE* out, int size, int my_np) {

	if (size > 90) {
		printf("ERROR!!! The method allow the size of matrix small than 90!!!\n");
		return 0;
	}
	/*int limit = get_devided_number_of_single_to_mul_sm(size,my_np);

	printf("Limit number of matrixes:%d\n", limit);*/

	if (my_np >= 1 && my_np < 1024) {
		return single_sm_inverse_gauss_gpu(out, size, my_np);
	}
	else {
		return more_sm_inverse_gauss_gpu(out, size, my_np);
	}
}
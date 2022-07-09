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
* MMIPB by Gauss
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

	extern __shared__ __DATA_TYPE mat[];
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


static
int single_sm_inverse_gauss_gpu(__DATA_TYPE* out, int size, int my_np) {

	int deviceSno = 0;
	cudaSetDevice(deviceSno);

	cudaEvent_t start, stop;
	cudaError_t cudaerror = cudaSuccess;

	DeviceInfo d;
	Device* ds = getDeviceByDeviceSno(deviceSno, d);
	int maxbytes = ds->sharedMemPerBlockOptin; // 65535 byte = 64 KB
	cudaFuncSetAttribute(Gauss_Jordan_Inverse, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
	int dy = min(floor(ds->sharedMemPerBlockOptin * 1.0 / (size * size * sizeof(__DATA_TYPE))), my_np);
	dy = min(1024 / size, dy); //dy->[1,1024]
	int matMaxSize = floor(sqrt(maxbytes * 1.0 / sizeof(__DATA_TYPE)));

	//Output some necessary infomation for remind you!
	int remain = my_np % dy;
	printf("%d matrix inverse works per block,MatMaxSize:%d\n", dy, matMaxSize);
	if (remain) {
		printf("Notice:There have %d last matrixes will not be inversed,You would better set MY_NP to multiples of %d!!\n", remain, dy);
	}


	CYW_TIMER timer;
	float time_elapsed = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	__DATA_TYPE* d_out;

	//cudaEventRecord(start, 0);
	gpuErrchk(cudaMalloc((void**)&d_out, my_np * size * size * sizeof(__DATA_TYPE)));
	//gpuErrchk(cudaMemcpy(d_out, out, my_np * size * size * sizeof(__DATA_TYPE), cudaMemcpyHostToDevice));

	cudaMemcpyToDeviceControl(d_out, out, my_np * size * size * sizeof(__DATA_TYPE),size,my_np);


	dim3 blocks(my_np / dy), threads(size, dy);
	timer.start_my_timer();
	cudaEventRecord(start, 0);
	Gauss_Jordan_Inverse << <blocks, threads, maxbytes >> > (d_out, size, dy);
	cudaEventRecord(stop, 0);
	timer.stop_my_timer();

	//gpuErrchk(cudaMemcpy(out, d_out, my_np * size * size * sizeof(__DATA_TYPE), cudaMemcpyDeviceToHost));
	cudaMemcpyToHostControl(d_out, out, my_np * size * size * sizeof(__DATA_TYPE),size);

	cudaEventSynchronize(start);    //Waits for an event to complete.
	cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务

	cudaEventElapsedTime(&time_elapsed, start, stop);    //计算时间差
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_out);

	printf("执行时间：%f(ms)\n", time_elapsed);
	printf("\n***GPU caosted time:");
	timer.print();
	cudaerror = cudaGetLastError();

	free_device_list(d.device);
	writeGPUResults(time_elapsed);
	if (cudaerror != cudaSuccess) {
		cudaCheck(cudaerror);
		return 0;
	}
	return 1;
}


static
int more_sm_inverse_gauss_gpu(__DATA_TYPE* out, int size, Device* d, int tasks, int my_np) {

	cudaSetDevice(d->index);

	cudaEvent_t start, stop;
	cudaError_t cudaerror = cudaSuccess;

	int maxbytes = d->sharedMemPerBlockOptin; // 65535 byte = 64 KB
	cudaFuncSetAttribute(Gauss_Jordan_Inverse, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);

	int sm = d->SMCount;
	int tasksPerStream = tasks / sm;
	int dis = tasksPerStream * size * size;

	//Compute number of works every block.
	int dy = min(floor(maxbytes * 1.0 / (size * size * sizeof(__DATA_TYPE))), tasksPerStream);
	dy = min(1024 / size, dy); //dy->[1,1024]
	int matMaxSize = floor(sqrt(maxbytes * 1.0 / sizeof(__DATA_TYPE)));

	float time_elapsed = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	__DATA_TYPE* d_out;

	cudaStream_t* stream = new cudaStream_t[sm];
	for (int i = 0; i < sm; i++) {
		gpuErrchk2(cudaStreamCreate(&stream[i]),d->index);
	}

	printf("%dth GPU, sm:%d,maxbytes:%d :%d matrix inverse works per block,MatMaxSize:%d\n",d->index, sm, maxbytes,dy, matMaxSize);
	int remainTasksPerSM = tasksPerStream % dy;
	if (remainTasksPerSM) {
		printf("MulSm Notice:There have %d last matrixes will not be inversed,it will happen every %d matrixes,You would better set tasks to multiples of %d!!\n", remainTasksPerSM, tasksPerStream, sm * dy);
	}

	gpuErrchk(cudaMalloc((void**)&d_out, tasks * size * size * sizeof(__DATA_TYPE)));
	cudaMemcpyToDeviceMulStreamsControl(d_out, out, tasksPerStream, sm, stream, size);

	dim3 blocks(tasksPerStream / dy), threads(size, dy);

	cudaEventRecord(start, 0);
	for (int k = 0; k < sm; k++) {
		Gauss_Jordan_Inverse << <blocks, threads, maxbytes, stream[k] >> > (d_out + k*dis, size, dy);
	}

	cudaEventRecord(stop, 0);
	cudaMemcpyToHostMulStreamsControl(d_out, out, tasksPerStream, sm, stream, size);

	cudaEventSynchronize(start);    //Waits for an event to complete.
	cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
	cudaEventElapsedTime(&time_elapsed, start, stop);    //计算时间差
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	gpuErrchk(cudaFree(d_out));

	printf("\n%dth GPU执行时间：%f(ms)\n", d->index, time_elapsed);
	cudaerror = cudaGetLastError();
	for (int i = 0; i < sm; ++i) {
		gpuErrchk2(cudaStreamDestroy(stream[i]),d->index);
	}
	writeGPUResults(time_elapsed);
	if (cudaerror != cudaSuccess) {
		cudaCheck(cudaerror);
		return 0;
	}
	return 1;
}



static
void asign_way(std::vector<std::thread> &threads, DeviceInfo &info,__DATA_TYPE * out,int size, int my_np) {

	int isFakeMemory = getConfigInt(USE_COPY_MATRIX_C);
	int way = getConfigInt(ALLOCATE_TASKS_WAY_C);
	int span = 0;
	int SMs = info.allDevicesSMCount;
	//int dis; 
	Devices d = info.device;
	if (way == ALLOCATE_BY_SM) {
		//dis = my_np / SMs;
		int start = 0, end = d->SMCount;
		for (int i = 0; i < info.deviceCount; i++) {
			//int tasks1 = dis * d->SMCount;
			int tasks = getCumulativeSumOfTasks(start,end,SMs,my_np);//解决余数问题
			//printf("%dth GPU have tasks:%d,\n", i, tasks);
			if (isFakeMemory == 0) {//not using fake memory.
				threads.push_back(std::thread(more_sm_inverse_gauss_gpu, out + span, size, d, tasks, my_np));
			}
			else {//using fake memory.
				threads.push_back(std::thread(more_sm_inverse_gauss_gpu, out + i * size * size, size, d, tasks, my_np));
			}
			span += tasks * size * size;
			d = d->next;
			start = end;
			if(d!=NULL)end += d->SMCount;
			else break;
		}
	}
	else if (way == ALLOCATE_BY_GLOBAL_MEMORY) {

	}
	else if (way == ALLOCATE_BY_AVERAGE) {
		//dis = my_np / info.deviceCount;//每个gpu处理的矩阵数量间隔
		for (int i = 0; i < info.deviceCount; i++) {
			//int tasks1 = dis;// 每个gpu处理的矩阵数量
			int tasks = getAmountOfTasks(i, info.deviceCount, my_np);
			//printf("%dth GPU have tasks:%d,\n", i, tasks);
			if (isFakeMemory == 0) {//not using fake memory.
				threads.push_back(std::thread(more_sm_inverse_gauss_gpu, out + span, size, d, tasks, my_np));
			}
			else {//using fake memory.
				threads.push_back(std::thread(more_sm_inverse_gauss_gpu, out + i * size * size, size, d, tasks, my_np));
			}
			span += tasks * size * size;
			d = d->next;
			if (d == NULL) break;
		}

	}
	else if (way == ALLOCATE_BY_CUSTOM) {
		float* rate = getConfigFloatArray(ALLOCATE_RATE_C, info.deviceCount);
		for (int i = 0; i < info.deviceCount; i++) {
			int tasks = round( rate[i] * my_np);// 每个gpu处理的矩阵数量
			printf("%dth GPU have tasks:%d,\n", i, tasks);
			global_memory_check(size,tasks, d);// 需要做内存检测，防止全局内存溢出。
			if (isFakeMemory == 0) {//not using fake memory.
				threads.push_back(std::thread(more_sm_inverse_gauss_gpu, out + span, size, d, tasks, my_np));
			}
			else {//using fake memory.
				threads.push_back(std::thread(more_sm_inverse_gauss_gpu, out + i * size * size, size, d, tasks, my_np));
			}
			span += tasks * size * size;
			d = d->next;
			if (d == NULL) break;
		}
	}
}


static
void asign_tasks(__DATA_TYPE* out, int size, int my_np) {
	//get information of each GPU
	DeviceInfo info;
	GetDeviceInfo(info);
	writeGPUResults("{");
	if (info.deviceCount <= 0) {
		printf("\nThere is no more GPU devices avalible! please add a GPU device with compute ability 6.5+ at least!!!!\n");
		return;
	}
	//build threads
	std::vector<std::thread> threads;
	//asign tasks acording different ways
	asign_way(threads, info, out, size, my_np);
	for (auto& th : threads) {
		th.join();
	}
	free_device_list(info.device);
	writeGPUResults("} ");
}


int my_more_gpu_for_single_block_mul_gauss_inverse_gpu(__DATA_TYPE* out, int size,int my_np) {

	cudaError_t cudaerror = cudaSuccess;
	CYW_TIMER timer;
	timer.init_my_timer();
	timer.start_my_timer();

	if (size > 90) {
		printf("ERROR!!! The method allow the size of matrix small than 90!!!\n");
		return 0;
	}

	if (my_np >= 1 && my_np < 3068) {
		single_sm_inverse_gauss_gpu(out, size, my_np);
	}
	else {
		asign_tasks(out, size, my_np);
	}

	timer.stop_my_timer();
	timer.print();
	cudaerror = cudaGetLastError();
	if (cudaerror != cudaSuccess) {
		cudaCheck(cudaerror);
		return 0;
	}
	return 1;
}
/*
 *
 * Copyright (C) 2020-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

#include "def.h"

/**
*
*get infomation of GPU
*/
void GetDeviceInfo(DeviceInfo& deviceInfo) {

	Devices d;
	int deviceCount;
	d = (Device*)malloc(sizeof(Device));
	Device *d1 = d,*d2;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	d1->SMCount = deviceProp.multiProcessorCount;
	d1->index = 0;
	d1->totalGlobalMem = deviceProp.totalGlobalMem;
	d1->sharedMemPerBlockOptin = deviceProp.sharedMemPerBlockOptin;

	cudaGetDeviceCount(&deviceCount);

	deviceInfo.deviceCount = deviceCount;
	deviceInfo.allDevicesSMCount = d1->SMCount;

	for (int i = 1; i < deviceCount; ++i) {
		d2 = (Device*)malloc(sizeof(Device));
		cudaGetDeviceProperties(&deviceProp, i);
		d2->SMCount = deviceProp.multiProcessorCount;
		d2->sharedMemPerBlockOptin = deviceProp.sharedMemPerBlockOptin;
		deviceInfo.allDevicesSMCount += d2->SMCount;
		d2->index = i;
		d2->totalGlobalMem = deviceProp.totalGlobalMem;
		d1->next = d2;
		d1 = d2;
	}
	d1->next = NULL;
	deviceInfo.device = d;
}

/// <summary>
/// Free List memory 
/// </summary>
void free_device_list(Device* d) {
	if(d->next)	free_device_list(d->next);
	if(d) free(d);
}

void CheckGPUInfo() {
	DeviceInfo d;
	GetDeviceInfo(d);
	Devices ds = d.device;
	printf("\n*****Checking GPU Information*****\n");
	printf("Number of devices:%d\n", d.deviceCount);
	printf("Number of sm in all devices:%d\n", d.allDevicesSMCount);
	int MY_N = getConfigInt("MY_N");
	int MY_NP = getConfigInt("MY_NP");
	double allGlobalMemory=0;
	int maxSM = 0;
	Device* device;
	if (d.deviceCount <= 0 || !ds) {
		printf("No available GPU!!!!\n");
		exit(0);
	}

	while (ds) {
		printf("%dth GPU has %d Streams,%.2f GB global memory. %.2f kb shared memory per block\n",
			ds->index, ds->SMCount, ds->totalGlobalMem*1.0/1024/1024/1024,ds->sharedMemPerBlockOptin *1.0/1024);
		allGlobalMemory += ds->totalGlobalMem;

		if (ds->SMCount > maxSM) {
			maxSM = ds->SMCount;
			device = ds;
		}

		ds = ds->next;
	}

	double need = sizeof(__DATA_TYPE) * MY_N * MY_N * MY_NP;
	int res = getConfigInt(ALLOCATE_TASKS_WAY_C);
	if (res == ALLOCATE_BY_SM) {
		printf("config set ALLOCATE_TASKS_WAY = ALLOCATE_BY_SM\n");
		if ((maxSM * 1.0 / d.allDevicesSMCount) * need > device->totalGlobalMem) {
			printf("%dth GPU global memory is not enough!!!\n",device->index);
			printf("The pragram is failed because of lacking global memory!!!\n");
			exit(0);
		}
	}
	else if (res == ALLOCATE_BY_GLOBAL_MEMORY) {
		printf("config set ALLOCATE_TASKS_WAY = ALLOCATE_BY_GLOBAL_MEMORY\n");
		if (allGlobalMemory < need) {
			printf("GPU global memory is not enough!!!\n");
			printf("The pragram is failed because of lacking global memory!!!\n");
			exit(0);
		}
	}
	else if (res == ALLOCATE_BY_CUSTOM) {
		printf("config set ALLOCATE_TASKS_WAY = ALLOCATE_BY_CUSTOM\n");
		float* rate = getConfigFloatArray(ALLOCATE_RATE_C, d.deviceCount);
		float sum = 0.0;
		for (int i = 0; i < d.deviceCount; i++) {
			if (rate[i] >= 0 && rate[i] <= 1) {
				sum += rate[i];
				printf("%lf ", rate[i]);
			}

			else {
				printf("Params 'ALLOCATE_RATE'in config.ini is not correct because of lacking of params.\n");
				exit(0);
			}
		}
		printf("\n");
		if (sum != 1.0) {
			printf("Params 'ALLOCATE_RATE'in config.ini is not correct,The sum of elements is 1.!!!\n");;
			exit(0);
		}

	}

	free_device_list(d.device);
}


Device* getDeviceByDeviceSno(int deviceSno) {
	DeviceInfo d;
	GetDeviceInfo(d);
	Device* ds = d.device;
	while (ds) {
		if (ds->index == deviceSno)
			break;
		else ds = ds->next;
	}
	return ds;
}

Device* getDeviceByDeviceSno(int deviceSno, DeviceInfo &d) {
	GetDeviceInfo(d);
	Device* ds = d.device;
	while (ds) {
		if (ds->index == deviceSno)
			break;
		else ds = ds->next;
	}
	return ds;
}
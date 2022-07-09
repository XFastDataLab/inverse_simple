/*
 *
 * Copyright (C) 2020-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

/*
* A tool for processing huge inverse matrixes in multiple gpus.
* It mainly focus on how to assigns tasks on multiple gpus.
*/
#include "def.h"


/// <summary>
/// the length of (end-start) more than 0,in other words , it is 1 at least.
/// </summary>
int getCumulativeSumOfTasks(int start, int end, int n, int sum) {
	int span = sum / n;
	int remain = sum % n;
	if (start <= remain && end <= remain) {
		return (span + 1) * (end - start);
	}
	else if (start <= remain && end <= n) {
		return (remain - start) * (span + 1) + (end - remain) * span;
	}
	else if (start <= n && end <= n) {
		return (end - start) * span;
	}
	return 0;
}


void global_memory_check(int size,int tasks, Device* d) {
	size_t size_tasks = tasks * size * size * sizeof(__DATA_TYPE);

	if (size_tasks > d->totalGlobalMem - 1 * 1024 * 1024 * 1024) {
		printf("%dth GPU needs %.5lf GB global memory\n", d->index, size_tasks * 1.0 / 1024 / 1024 / 1024);
		printf("%dth GPU global memory is not enought, please checking\n", d->index);
		exit(0);
	}
}


int get_devided_number_of_single_to_mul_sm(int size, int my_np) {
	int deviceSno = 0;
	DeviceInfo d;
	GetDeviceInfo(d);
	Device* ds = getDeviceByDeviceSno(deviceSno, d);
	int maxbytes = ds->sharedMemPerBlockOptin; // 65535 byte = 64 KB
	int dy = min(floor(ds->sharedMemPerBlockOptin * 1.0 / (size * size * sizeof(__DATA_TYPE))), my_np);
	dy = min(1024 / size, dy); //dy->[1,1024]
	free_device_list(d.device);
	return ceil(dy * 65536 * 1.0 / 35);//35 颗寄存器。
}

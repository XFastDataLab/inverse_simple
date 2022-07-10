/*
 *
 * Copyright (C) 2020-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

#pragma once
#ifndef DEVICE_INFO_H
#define DEVICE_INFO_H

/*
 Devices of specific information
*/
typedef struct Device {
	int index;
	int SMCount;
	size_t totalGlobalMem;
	size_t sharedMemPerBlockOptin; /**< Per device maximum shared memory per block usable by special opt in */
	int multiProcessorCount;
	struct Device* next;
}Device, * Devices;

typedef struct DeviceInfo {
	int deviceCount;
	int allDevicesSMCount;
	/*int minSizeOfMatrixSupportShared;
	int maxSizeOfMatrixSupportShared;*/
	struct Device* device;
}DeviceInfo;

void GetDeviceInfo(DeviceInfo& deviceInfo);

void free_device_list(Device* d);

void CheckGPUInfo();

Device* getDeviceByDeviceSno(int deviceSno);

Device* getDeviceByDeviceSno(int deviceSno, DeviceInfo& d);
#endif // !DEVICE_INFO_H


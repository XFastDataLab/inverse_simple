/*
 *
 * Copyright (C) 2020-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

#include "def.h"


void test_data1() {
	int MY_N = getConfigInt("MY_N");
	int MY_NP = getConfigInt("MY_NP");


	int mat_sizes[] = {
		2,3,4,5,6,7,8,9,10,
		11,12,13,14,15,16,17,18,19,20,
		21,22,23,24,25,26,27,28,29,30,
		31,32,33,34,35,36,37,38,39,40,
		41,42,43,44,45,46,47,48,49,50,
		51,52,53,54,55,56,57,58,59,60,
		61,62,63,64,65,66,67,68,69,70
	};

	int sizeOfmatSizes = 69;

	int mat_numbers[] = {
		1, 8 , 32, 128, 512, 1024, 2048,
		4096, 5120, 8192, 10240, 15360,20480,25600,30720,35840,
		40960,46080,51240,56320,61440,66560,71680,76800,81920
	};

	int sizeOfmatNum = 25;


	clearCPUResults();
	clearGPUResults();
	clearCheckedInfo();

	for (int i = 0; i < sizeOfmatNum; i++) {
		writeCPUResults(mat_numbers[i]);
		writeGPUResults(mat_numbers[i]);
	}
	writeCPUResults(0, true);
	writeGPUResults(0, true);

	CheckGPUInfo();
	//type of matrix: definite/random/triangleLow/triangleUp
	for (int i = 0; i < sizeOfmatSizes; i++) {
		setConfigInt("MY_N", mat_sizes[i]);
		printf("i is %d\n", mat_sizes[i]);
		for (int j = 0; j < sizeOfmatNum; j++) {
			CheckMemoryInfo(mat_sizes[i], mat_numbers[j]);
			setConfigInt("MY_NP", mat_numbers[j]);
			test_cublas(mat_sizes[i], mat_numbers[j]);
		}
		writeCPUResults(0, true);
		writeGPUResults(0, true);
	}
}

int main() {
	
	//warm up
	test_cublas(8, 2048);

	test_data1();
	return 0;
}
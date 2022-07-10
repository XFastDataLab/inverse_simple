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
		31,32,
	};
	int sizeOfmatSizes = 31;

	/*int mat_sizes[] = {
		2,3,4,5,6,7,8,9,10,
		11,12,13,14,15,16,17,18,19,20,
		21,22,23,24,25,26,27,28,29,30,
		31,32,33,34,35,36,37,38,39,40,
		41,42,43,44,45,46,47,48,49,50,
		51,52,53,54,55,56,57,
	};

	int sizeOfmatSizes = 56;*/

	/*int mat_sizes[] = {
		2,3,4,5,6,7,8,9,10,
		11,12,13,14,15,16,17,18,19,20,
		21,22,23,24,25,26,27,28,29,30,
		31,32,33,34,35,36,37,38,39,40,
		41,42,43,44,45,46,47,48,49,50,
		51,52,53,54,55,56,57,58,59,60,
		61,62,63,64,65,66,67,68,69,70
	};

	int sizeOfmatSizes = 69;*/

	int mat_numbers[] = {
		163840, 327680, 491520,655360, 819200, 983040
	};
	int sizeOfmatNum = 6;

	/*int mat_numbers[] = {
		4096, 5120, 8192, 10240, 15360,20480,25600,30720,35840,
		40960,46080,51240,56320,61440,66560,71680,76800,81920
	};

	int sizeOfmatNum = 18;*/


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
	//return 0;
	//type of matrix: definite/random/triangleLow/triangleUp
	// //测试算法2，batched算法4
	for (int i = 0; i < sizeOfmatSizes; i++) {
		setConfigInt("MY_N", mat_sizes[i]);
		printf("i is %d\n", mat_sizes[i]);
		for (int j = 0; j < sizeOfmatNum; j++) {
			//CheckMemoryInfo(mat_sizes[i], mat_numbers[j]);
			setConfigInt("MY_NP", mat_numbers[j]);
			printf("j is %d\n", mat_numbers[j]);
			//test_algorithm2(mat_sizes[i], mat_numbers[j], "definite");
			test_batched_single_block_mul_gauss_inverse_gpu(mat_sizes[i], mat_numbers[j], "definite");
			//test_cublas(mat_sizes[i], mat_numbers[j], "definite");
			//printf("\n\n*****TEST SINGLE BLOCK MULTIBL GAUSS INVERSE ON MULTIPLE CPUs!!!*****\n");
			//printf("\n\n*****size:%d,number of matrix is %d\n", mat_sizes[i], mat_numbers[j]);
			//test_gauss_on_cpu(mat_sizes[i], mat_numbers[j], "definite");
		}
		writeCPUResults(0, true);
		writeGPUResults(0, true);
	}

	//算法3：多GPU测试算法2
	/*for (int i = 0; i < sizeOfmatSizes; i++) {
		setConfigInt("MY_N", mat_sizes[i]);
		for (int j = 0; j < sizeOfmatNum; j++) {
			CheckMemoryInfo(mat_sizes[i], mat_numbers[j]);
			setConfigInt("MY_NP", mat_numbers[j]);
			test_more_gpu_for_single_block_mul_gauss_inverse_gpu(mat_sizes[i], mat_numbers[j], "definite");
		}
		writeCPUResults(0, true);
		writeGPUResults(0, true);
	}*/

	//test_cublas(mat_sizes[2], mat_numbers[12], "definite");

	//test_single_block_mul_gauss_inverse_gpu(mat_sizes[2], mat_numbers[12], "definite");
}

int main() {
	
	int MY_N = getConfigInt("MY_N");
	int MY_NP = getConfigInt("MY_NP");

	int mat_sizes[] = {
		2
	};
	int sizeOfmatSizes = 1;


	int mat_numbers[] = {
		1,100,1024,10240,102400,1024000,10240000
	};
	int sizeOfmatNum =7;

	
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
	//return 0;
	//type of matrix: definite/random/triangleLow/triangleUp
	// //测试算法2，batched算法4
	for (int i = 0; i < sizeOfmatSizes; i++) {
		setConfigInt("MY_N", mat_sizes[i]);
		printf("i is %d\n", mat_sizes[i]);
		for (int j = 0; j < sizeOfmatNum; j++) {
			//CheckMemoryInfo(mat_sizes[i], mat_numbers[j]);
			setConfigInt("MY_NP", mat_numbers[j]);
			test_fun_2(mat_sizes[i], mat_numbers[j], "definite");
			test_gauss_on_cpu(mat_sizes[i], mat_numbers[j], "definite");
		}
		writeCPUResults(0, true);
		writeGPUResults(0, true);
	}


	return 0;
}
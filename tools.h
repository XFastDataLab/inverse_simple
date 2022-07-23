/*
 *
 * Copyright (C) 2020-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

#ifndef MY_TOOLS_H
#define MY_TOOLS_H

#define cudaCheck(ans) do{if(ans != cudaSuccess){fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(ans),  __FILE__, __LINE__); exit(EXIT_FAILURE);} }while(false)

#define gpuErrchk(ans) { tools_gpuAssert((ans), __FILE__, __LINE__); }

#define gpuErrchk2(ans,index) { tools_gpuAssert(index,(ans), __FILE__, __LINE__); }

/**
 * CUDA Error types
 */
typedef enum cudaError cudaError_t;

/*
    Debug output
*/
void tools_gpuAssert(cudaError_t code, const char *file, int line);

void tools_gpuAssert(int index, cudaError_t code, const char* file, int line);

/*
  Print a Matrix with with N x N dimension
*/
void tools_print_matrix(__DATA_TYPE * matrix, int N);

void tools_print_array(__DATA_TYPE* array, int N);

/*
  Print a Matrix more beautiful 
*/
void tools_WAprint(int size_of_one_side, __DATA_TYPE * matrix);

/*
  checks for zero with a window of e^-5
*/
int tools_zero(__DATA_TYPE f);

/*
  simply check the bit patterns.. hope that the gpu uses the same precision as the cpu
*/
int tools_is_equal(__DATA_TYPE * a, __DATA_TYPE * b, int size);

bool is_equal(__DATA_TYPE a, __DATA_TYPE b);

__DATA_TYPE* random_matrix_generate_by_matlab(int n, int my_np, std::string path);

float** random_matrix_generate_by_matlab2(int n, int my_np, std::string path);

__DATA_TYPE* random_matrix_generate_by_matlab(int n, int my_np);

void writeCPUResults(std::string s);

void writeCPUResults(double time, bool isNextLine = false);

void writeGPUResults(std::string s);

void writeCheckedInfo(int size, int my_np, int index, __DATA_TYPE realVal, __DATA_TYPE testVal);

void clearCPUResults();

void clearGPUResults();

void clearCheckedInfo();

void writeGPUResults(double time, bool isNextLine = false);

std::string num2str(int num);

int str2int(std::string s);

__DATA_TYPE readInversedMatrix(int size, std::string type);

#endif
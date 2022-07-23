/*
 *
 * Copyright (C) 2020-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

#pragma once

//unified data type
typedef  float __DATA_TYPE;

#include "windows.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <vector>
#include <thread>

//Cuda libraries
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_profiler_api.h>
#include "device_launch_parameters.h"
#include "device_functions.h"


//Fixes for windows CC
//#include "wingetopt.h"

//Tools
#include "tools.h"
#include "tools_for_matrix_inverse.h"
#include "config_params.h"
#include "host_info.h"
#include "device_info.h"
#include "tools_for_more_gpu.h"


//Src libraries

#include "cyw_timer.h"
#include "test.h"




#include "cublas_inv_matrix.cuh"






#define min(a,b)            (((a) < (b)) ? (a) : (b))







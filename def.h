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



//Fixes for windows CC
//#include "wingetopt.h"

//Tools
#include "tools.h"
#include "host_info.h"

//Src libraries

#include "cyw_timer.h"
#include "test.h"

#include "gauss_inverse_cpu.h"



#define min(a,b)            (((a) < (b)) ? (a) : (b))







/*
 *
 * Copyright (C) 2020-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

#pragma once
#ifndef CONFIG_PARAMS_H
#define CONFIG_PARAMS_H

#include<string>

using namespace std;

/// <summary>
/// set it to true when host memory is not enough
/// </summary>
extern const string USE_COPY_MATRIX_C;
extern const string ALLOCATE_TASKS_WAY_C;
extern const string ALLOCATE_RATE_C;
extern const string MY_AMOUNT_MATRIX_C;

const enum ALLOCATE_TASKS_WAY_ENUM {
	ALLOCATE_BY_SM = 1,
	ALLOCATE_BY_GLOBAL_MEMORY = 2,
	ALLOCATE_BY_AVERAGE = 3,
	ALLOCATE_BY_CUSTOM = 4,
};

/// <summary>
/// Set value of params in 'config.ini'.
/// </summary>
/// <param name="str"></param>
/// <param name="value"></param>
void setConfigString(string key, string value);


LPSTR getConfigString(string key);

int getConfigInt(string key);

void setConfigInt(string key, int val);

void setConfigFloat(string key, float time_elapsed);

float getConfigFloat(string key);

void addConfigFloat(string key, float time_inscreased);

void setConfigFloatArray(string key, float* value, int n);

float* getConfigFloatArray(string key, int n);

#endif // !CONFIG_PARAMS_H


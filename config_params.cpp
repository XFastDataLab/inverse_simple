/*
 *
 * Copyright (C) 2020-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

#include "def.h"
#include<sstream>
#include<iomanip>

const string USE_COPY_MATRIX_C		=	"USE_COPY_MATRIX";
const string ALLOCATE_TASKS_WAY_C	=	"ALLOCATE_TASKS_WAY";
const string ALLOCATE_RATE_C		=	"ALLOCATE_RATE";
const string MY_AMOUNT_MATRIX_C		=	"MY_AMOUNT_MATRIX";



using namespace std;
void setConfigString(string key, string value) {
	WritePrivateProfileString("CONFIG", key.c_str(), value.c_str(), "./config.ini");
}


LPSTR getConfigString(string key) {
	LPSTR value = new char[12];
	GetPrivateProfileString("CONFIG", key.c_str(), NULL,value,12, "./config.ini");
	return value;
}

int getConfigInt(string key) {

	return GetPrivateProfileInt("CONFIG", key.c_str(), 0, "./config.ini");
}

void setConfigInt(string key, int val) {
	ostringstream foo;
	foo << val;
	WritePrivateProfileString("CONFIG", key.c_str(), foo.str().c_str(), "./config.ini");
}


void setConfigFloat(string key, float time_elapsed) {
	ostringstream foo;
	foo <<  std::setprecision(15)<<time_elapsed;
	WritePrivateProfileString("CONFIG", key.c_str(), foo.str().c_str(), "./config.ini");
}

float getConfigFloat(string key) {
	LPSTR value = new char[21];
	GetPrivateProfileString("CONFIG", key.c_str(), NULL, value, 21, "./config.ini");
	istringstream is(value);
	float v;
	is >> v;
	return v;
}


void addConfigFloat(string key, float time_inscreased) {
	float elapse = getConfigFloat(key);
	elapse += time_inscreased;
	setConfigFloat(key, elapse);
}


void setConfigFloatArray(string key, float *value, int n) {

	ostringstream foo;
	for (int i = 0; i < n; i++) {
		foo << value[i] << " ";
	}

	WritePrivateProfileString("CONFIG", key.c_str(), foo.str().c_str(), "./config.ini");
}


float* getConfigFloatArray(string key, int n) {

	LPSTR value = new char[n*10];
	GetPrivateProfileString("CONFIG", key.c_str(), NULL, value, n*10, "./config.ini");
	istringstream is(value);
	float* v = new float[n];
	for (int i = 0; i < n; i++) {
		is >> v[i];
	}
	return v;
}
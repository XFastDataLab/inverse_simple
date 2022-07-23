/* 
 * Tools used for debugging printing and more
*/
#include "def.h"

/*
    Debug output
*/
void tools_gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
   }
}

void tools_gpuAssert(int index,cudaError_t code, const char* file, int line)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "%dth GPUassert: %s %s %d\n",index, cudaGetErrorString(code), file, line);
	}
}

static void printInfo(__DATA_TYPE data) {
	if (is_same<__DATA_TYPE, double>::value) {
		printf("%6.15lf ", data);
	}
	else if (is_same<__DATA_TYPE, float>::value) {
		printf("%f ", data);
	}
}

/*
  Print a Matrix with with N x N dimension
*/
void tools_print_matrix(__DATA_TYPE* matrix, int N){
	int i;
	for(i = 0; i < N; ++i){
		int j;
		for(j = 0;j < N; ++j){
			printInfo(matrix[i * N + j]);
		}
		printf("\n\r");
	}
	printf("\n\r");
}

void tools_print_array(__DATA_TYPE* array, int N) {
	int i;
	for (i = 0; i < N; ++i) {
		printInfo(array[i]);
	}
	printf("\n\r");
}

/*
  Print a Matrix more beautiful 
*/
void tools_WAprint(int size_of_one_side, __DATA_TYPE* matrix){
	printf("WA output form:\n");
	printf("inverse {");
	for(int x = 0; x < size_of_one_side; x++) {
		printf("{");
		for(int y = 0; y < size_of_one_side; y++) {
			printf("%lf", matrix[x*size_of_one_side + y]);
			if(y != size_of_one_side-1)
				printf(",");
		}
		printf("}");
		if(x != size_of_one_side-1)
			printf(",");
	}
	printf("}\n");
}

/*
  checks for zero with a window of e^-5
*/
int tools_zero(__DATA_TYPE f){
	if(fabs(f) < 10e-12){
		return 1;
	}
  return 0;
}



/*
  simply check the bit patterns.. hope that the gpu uses the same precision as the cpu
*/
int tools_is_equal(__DATA_TYPE * a, __DATA_TYPE * b, int size=1){
	int i;
	int ret = 1;
	for(i = 0;i < size;i++){
		double dis = fabs((a[i] - b[i]));
		if (is_same<double, __DATA_TYPE>::value && dis >0.0000000000001) {
			ret = 0;
		}
		else if (is_same<float, __DATA_TYPE>::value&&dis < 0.00001) {
			ret = 0;
		}
	}
	return ret;
}

bool is_equal(__DATA_TYPE a, __DATA_TYPE b) {
	double dis = fabs((a - b));
	if (is_same<double, __DATA_TYPE>::value && dis < 0.0000000000001) {
		return 1;
	}
	else if (dis < 0.00001) {
		return 1;
	}
	return 0;
}


string num2str(int num) {
	std::stringstream ss;
	ss << num;
	return ss.str();
}

int str2int(string s) {
	int num;
	std::stringstream ss(s);
	ss >> num;
	return num;
}


__DATA_TYPE* random_matrix_generate_by_matlab(int n,int my_np, std::string path) {
	int size = n * n;
	__DATA_TYPE* d_mat;
	int res = getConfigInt(USE_COPY_MATRIX_C);

	//get number of devices
	DeviceInfo info;
	GetDeviceInfo(info);
	int deviceCounts = info.deviceCount;

	if (res == 0) d_mat = (__DATA_TYPE*)malloc(sizeof(__DATA_TYPE) * size * my_np);
	else d_mat = (__DATA_TYPE*)malloc(sizeof(__DATA_TYPE) * size * deviceCounts);

	FILE* file;
	fopen_s(&file, path.c_str(), "r");
	if (file == NULL) {
		cout << "random_matrix_generate_by_matlab ";
		cout << path;
		cout << " File is not existed!!!" << endl;
		exit(0);
	}
	int r;
	for (int i = 0; i < size; i++) {
		r = 0;
		if (is_same<double, __DATA_TYPE>::value) {
			r = fscanf(file, "%lf", d_mat + i);
		}
		else if (is_same<float, __DATA_TYPE>::value) {
			r = fscanf(file, "%f", d_mat + i);
		}
		
		if (r == EOF || r == 0) {
			printf("Read file of data1.txt ERROR!!!!!");

		}
	}


	if (res == 0) {
		for (int i = size; i < size * my_np; i++) {
			d_mat[i] = d_mat[i % size];
		}
	}
	else {
		for (int i = size; i < size * deviceCounts; i++) {
			d_mat[i] = d_mat[i % size];
		}
	}

	fclose(file);
	free_device_list(info.device);
	return d_mat;
}

float** random_matrix_generate_by_matlab2(int n, int my_np, std::string path) {
	int res = getConfigInt(USE_COPY_MATRIX_C);

	cout << 1 << endl;
	//get number of devices
	DeviceInfo info;
	GetDeviceInfo(info);
	int deviceCounts = info.deviceCount;
	cout << 2 << endl;
	int size = n * n;
	float** d_mat = new float* [my_np];
	cout << 3 << endl;
	if (res == 0) {
		for (int i = 0; i < my_np; i++) {
			cudaMalloc((void**)&d_mat[i], sizeof(float) * size);
			//d_mat[i] = (float*)malloc(sizeof(float) * size);
		}
	}
	else {
		for (int i = 0; i < deviceCounts; i++) {
			cudaMalloc((void**)&d_mat[i], sizeof(float) * size);
			//d_mat[i] = (float*)malloc(sizeof(float) * size);
		}
	}
	cout << 4 << endl;

	FILE* file = fopen(path.c_str(), "r");
	int r;
	cout << 5 << endl;
	for (int i = 0; i < size; i++) {
		r = 0;
		float temp;
		double temp1;
		cout << i << endl;
		if (is_same<double, __DATA_TYPE>::value) {
			r = fscanf(file, "%lf", &temp);
			cudaMemcpy(&d_mat[0][i], &temp, sizeof(float), cudaMemcpyHostToDevice);
		}
		
		else if (is_same<float, __DATA_TYPE>::value) {
			r = fscanf(file, "%f", &temp1);
			cudaMemcpy(&d_mat[0][i], &temp1, sizeof(double), cudaMemcpyHostToDevice);
		}
		cout << i << endl;
		if (r == EOF || r == 0) {
			printf("Read file of data1.txt ERROR!!!!!");
			cout << i << endl;
		}
	}
	cout << 6 << endl;
	if (res == 0) {
		for (int i = 1; i < my_np; i++) {
			//memcpy(d_mat[i], d_mat[0], sizeof(float) * size);
			cudaMemcpy(d_mat[i], d_mat[0], sizeof(float) * size, cudaMemcpyDeviceToDevice);
		}
	}
	else {
		for (int i = 1; i < deviceCounts; i++) {
			//memcpy(d_mat[i], d_mat[0], sizeof(float) * size);
			cudaMemcpy(d_mat[i], d_mat[0], sizeof(float) * size, cudaMemcpyDeviceToDevice);
		}
	}
	cout << 7 << endl;
	free_device_list(info.device);
	return d_mat;
}

__DATA_TYPE* random_matrix_generate_by_matlab(int n, int my_np) {
	return random_matrix_generate_by_matlab(n, my_np, string("./data/definite/").append(num2str(n)).append("/ data1.txt"));
}


void writeCPUResults(string s) {
	fstream fs;
	fs.open("./results_cpu.txt", std::fstream::app);
	fs << s;
	fs.close();
}

void writeCPUResults(double time, bool isNextLine) {
	fstream fs;
	fs.open("./results_cpu.txt",std::fstream::app);

	if (isNextLine) {
		fs << endl;
	}
	else {
		fs.setf(std::ios::fixed, std::ios::floatfield);
		fs.precision(10);
		fs << time;
		fs << " ";
	}
	fs.close();
}

void writeGPUResults(string s) {
	fstream fs;
	fs.open("./results_gpu.txt", std::fstream::app);
	fs << s;
	fs.close();
}

void writeGPUResults(double time, bool isNextLine) {
	fstream fs;
	fs.open("./results_gpu.txt", std::fstream::app);

	if (isNextLine) {
		fs << endl;
	}
	else {
		fs.setf(std::ios::fixed, std::ios::floatfield);
		fs.precision(10);
		fs << time;
		fs << " ";
	}
	fs.close();
}

void writeCheckedInfo(int size, int my_np, int index, __DATA_TYPE realVal, __DATA_TYPE testVal) {
	fstream fs("./checkedInfo.txt", std::fstream::app);
	fs << " size:" << size << " my_np:" << my_np << " index:" << index << "::";
	fs.setf(std::ios::fixed, std::ios::floatfield);
	fs.precision(15);
	fs << "realVal:" << realVal << " testVal:" << testVal << ".";
	if (is_equal(realVal, testVal)) {
		fs << "result:YES" << endl;;
	}
	else {
		fs << "result:NO" << endl;;
	}
}

void clearFileContent(string path) {
	fstream fs;
	fs.open(path, std::ios::out | std::fstream::trunc);
	if (fs) fs.close();
}

void clearCPUResults() {
	clearFileContent("./results_cpu.txt");
}

void clearGPUResults() {
	clearFileContent("./results_gpu.txt");
}

void clearCheckedInfo() {
	clearFileContent("./checkedInfo.txt");
}


__DATA_TYPE readInversedMatrix(int size, std::string type) {

	__DATA_TYPE val;
	string path = string("./data/").append(type).append("/").append(num2str(size)).append("/data1.txt");
	fstream fs(path);
	if (fs.is_open()) {
		string str;
		getline(fs, str);
		getline(fs, str);
		getline(fs, str);
		fs >> val;
		fs.close();
		return val;
	}
	else {
		cout << "readInversedMatrix";
		cout << path;
		cout << " File is not existed!!!" << endl;

		return -1;
	}

}
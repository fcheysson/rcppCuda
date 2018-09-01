/* CUDA API header files*/
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// CUBLAS library
#include "cublas_v2.h"

extern "C"
void amax(double *x, int *index, int *size) {
	// Initialise a few things
	cudaError_t cudaStat; // cudaMalloc status
	cublasStatus_t stat; // CUBLAS functions status
	cublasHandle_t handle; // CUBLAS context
	
	// Copy passed argument on the device
	double *xd;
	cudaStat = cudaMalloc((void**) &xd, *size * sizeof(*x));	// Memory alloc for xd
	stat = cublasCreate(&handle);	// Initialize CUBLAS context
	stat = cublasSetVector(*size, sizeof(*x), x, 1, xd, 1);	//x -> xd

	// Run the CUBLAS function
	stat = cublasIdamax(handle, *size, xd, 1, index);
	
	cudaFree(xd); // free memory
	cublasDestroy(handle); // destroy CUBLAS context
}

extern "C"
void amax_u(double *x, int *index, int *size) {
	cublasHandle_t handle; // CUBLAS context

	// Copy passed argument on the device
	double *xd;
	cudaMalloc((void**) &xd, *size * sizeof(*x));
	cudaMemcpy(xd, x, *size * sizeof(*x), cudaMemcpyHostToDevice);	// copy data to device
	
	// Run the CUBLAS function
	cublasCreate(&handle);	// Initialize CUBLAS context
	cublasIdamax(handle, *size, xd, 1, index);
	cudaDeviceSynchronize(); // Blocks until the device has completed all preceding requested tasks

	cudaFree(xd); // free memory
	cublasDestroy(handle); // destroy CUBLAS context
}
/* CUDA API header files*/
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void
matrixMult(const double *Md, const double *Nd, double *Pd, int size)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;

	if (row < size) {	// Don't do anything to the memory if we're above the size of the matrix
		if (col < size) {
			
			double Pvalue = 0;
			for (int k = 0; k < size; k++) {	
				// Elements of 2d-arrays are stored in column-major ordering (i.e. column by column)
				// This is a consequence of this code being called in R (where column-major ordering is the norm)
				// whereas C usually stores 2d-array in row-major ordering
				Pvalue += Md[k*size + row] * Nd[col*size + k];
			}
			Pd[col*size + row] = Pvalue;
			
		}
	}
}

extern "C"
void gmatrixMult(double *M, double *N, double *P, int *size)
{
	int memSize = *size * *size * sizeof(double);
	// Device Memory
	double *Md, *Nd, *Pd;
	// Define the execution configuration
	dim3 blockSize(32, 32, 1);
	dim3 gridSize(1, 1, 1);
	gridSize.x = (*size + blockSize.x - 1) / blockSize.x;
	gridSize.y = (*size + blockSize.y - 1) / blockSize.y;
	// Allocate output array
	cudaMalloc((void**)&Md, memSize);
	cudaMalloc((void**)&Nd, memSize);
	cudaMalloc((void**)&Pd, memSize);
	// copy data to device
	cudaMemcpy(Md, M, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(Nd, N, memSize, cudaMemcpyHostToDevice);
	// GPU matrix multiplication
	matrixMult<<<gridSize, blockSize>>>(Md, Nd, Pd, *size);
	// Copy output
	cudaMemcpy(P, Pd, memSize, cudaMemcpyDeviceToHost);
	cudaFree(Md);
	cudaFree(Nd);
	cudaFree(Pd);
}
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*Kernel*/
__global__ void vectorAdd(float a[], float b[], float c[], int N) {
   int index = blockDim.x * blockIdx.x + threadIdx.x;

   if (blockIdx.x < N && threadIdx.x < N) 
      c[index] = a[index] + b[index];
}

void vecAdd(float* A, float* B, float* C, int N)
{
	int size = N * sizeof(float);
	float *d_A, *d_B, *d_C;	

	cudaMalloc((void **) &d_A, size);
	cudaMalloc((void **) &d_B, size);
	cudaMalloc((void **) &d_C, size);

	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	
 	vectorAdd<<<N,N>>> (d_A, d_B, d_C, N);

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
 // Free device memory for A, B, C
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree (d_C);
}

void Print_vector(char name[], float A[], int s) {
   int i;

   printf("%s\n", name);
   for (i = 0; i < s; i++) {
      printf("%.1f ", A[i]);
      printf("\n");
   }  
}
void Read_vector(float A[], int s) {
   int i;

   for (i = 0; i < s; i++)
      scanf("%f", &A[i]);
}


int main(int argc, char* argv[]) {
   int N;
   int size_vector;
   float *dev_a, *dev_b, *dev_c;
   float *a, *b, *c;

   N = strtol(argv[1], NULL, 10);
   printf("size = %d", N);

   size_vector = N*sizeof(float);


   a = (float*) malloc(size_vector);
   b = (float*) malloc(size_vector);
   c = (float*) malloc(size_vector); 

   printf("vector A: \n");
   Read_vector(a, N);
   printf("vector B: \n");
   Read_vector(b, N);

   Print_vector("A =", a, N);
   Print_vector("B =", b, N);
   vecAdd(a,b,c,N);


   Print_vector("Result: ",c,N);
   free(a);
   free(b);
   free(c);

   return 0;
}
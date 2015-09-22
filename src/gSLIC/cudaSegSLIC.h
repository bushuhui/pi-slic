#ifndef __CUDA_SEG_SLIC__
#define  __CUDA_SEG_SLIC__

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudaDefines.h"

typedef struct
{
	float4 lab;
	float2 xy;
	int nPoints;
	int x1, y1, x2, y2;
} SLICClusterCenter;

__host__ void SLICImgSeg(int32_t* maskBuffer, float4* floatBuffer,
						 int nWidth, int nHeight, int nSegs,  
						 SLICClusterCenter* vSLICCenterList,
						 int listSize,
						 float weight);

__global__ void kInitClusterCenters(float4* floatBuffer, 
									int nWidth, int nHeight,
									SLICClusterCenter* vSLICCenterList);

__global__ void kIterateKmeans(int32_t* maskBuffer, float4* floatBuffer,
							   int nWidth, int nHeight, int nSegs, int nClusterIdxStride, 
							   SLICClusterCenter* vSLICCenterList, int listSize,
							   bool bLabelImg, float weight);

__global__ void kUpdateClusterCenters(float4* floatBuffer, int32_t* maskBuffer,
										  int nWidth, int nHeight, int nSegs,  
										  SLICClusterCenter* vSLICCenterList, int listSize);

void enforceConnectivity(int32_t* maskBuffer,int width, int height, int nSeg);

#endif

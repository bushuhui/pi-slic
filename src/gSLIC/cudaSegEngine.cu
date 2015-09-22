#include "cudaSegEngine.h"
#include "cudaUtil.h"
#include "cudaImgTrans.h"
#include "cudaSegSLIC.h"

#include <stdio.h>
#include <math.h>
#include <time.h>

using namespace std;

__device__ uchar4*      rgbBuffer;
__device__ float4*      floatBuffer;
__device__ int32_t*     maskBuffer;

int nWidth,nHeight,nSeg,nMaxSegs;
bool cudaIsInitialized=false;

__device__ SLICClusterCenter* vSLICCenterList;
bool slicIsInitialized=false;

__host__ void InitCUDA(int width, int height,int nSegment, SEGMETHOD eMethod)
{
	//for all methods
	if (!cudaIsInitialized)
	{
		nWidth=width;
		nHeight=height;

		cudaMalloc((void**) &rgbBuffer,width*height*sizeof(uchar4));
		cudaMalloc((void**) &floatBuffer,width*height*sizeof(float4));
        cudaMalloc((void**) &maskBuffer,width*height*sizeof(int32_t));

		cudaMemset(floatBuffer,0,width*height*sizeof(float4));
        cudaMemset(maskBuffer,0,width*height*sizeof(int32_t));

		nSeg=nSegment;
		cudaIsInitialized=true;
	}

	if (!slicIsInitialized)
	{
		// MaxSegs should be same on initializeFastSeg()@FastImgSeg.cpp
		nMaxSegs=(iDivUp(nWidth,BLCK_SIZE)*2)*(iDivUp(nHeight,BLCK_SIZE)*2);

		// the actual number of segments
		cudaMalloc((void**) &vSLICCenterList,nMaxSegs*sizeof(SLICClusterCenter));
		cudaMemset(vSLICCenterList,0,nMaxSegs*sizeof(SLICClusterCenter));
		slicIsInitialized=true;
	}
}

extern "C" __host__ void CUDALoadImg(unsigned char* imgPixels)
{
	if (cudaIsInitialized)
	{
		cudaMemcpy(rgbBuffer,imgPixels,nWidth*nHeight*sizeof(uchar4),cudaMemcpyHostToDevice);
	}
	else
	{
		return;
	}
}

__host__ void TerminateCUDA()
{
	if (cudaIsInitialized)
	{
		cudaFree(rgbBuffer);
		cudaFree(floatBuffer);
		cudaFree(maskBuffer);

		cudaIsInitialized=false;
	}

	if (slicIsInitialized)
	{
		cudaFree(vSLICCenterList);

		slicIsInitialized=false;
	}
}

__host__ void CudaSegmentation(SEGMETHOD eSegmethod, double weight)
{
	switch (eSegmethod)
	{
	case SLIC :

		Rgb2CIELab(rgbBuffer,floatBuffer,nWidth,nHeight);
		SLICImgSeg(maskBuffer,floatBuffer,nWidth,nHeight,nSeg,vSLICCenterList,nMaxSegs,(float)weight);

		break;

	case RGB_SLIC:

		Uchar4ToFloat4(rgbBuffer,floatBuffer,nWidth,nHeight);
		SLICImgSeg(maskBuffer,floatBuffer,nWidth,nHeight,nSeg,vSLICCenterList,nMaxSegs,(float)weight);

		break;

	case XYZ_SLIC:

		Rgb2XYZ(rgbBuffer,floatBuffer,nWidth,nHeight);
		SLICImgSeg(maskBuffer,floatBuffer,nWidth,nHeight,nSeg,vSLICCenterList,nMaxSegs,(float)weight);

		break;
	}

	cudaThreadSynchronize();
}

__host__ void CopyImgDeviceToHost( unsigned char* imgPixels, int width, int height)
{
	if (cudaIsInitialized)
	{
		cudaMemcpy(imgPixels,rgbBuffer,nHeight*nWidth*sizeof(uchar4),cudaMemcpyDeviceToHost);
	}
}

__host__ void CopyMaskDeviceToHost( int32_t* maskPixels)
{
	if (cudaIsInitialized)
	{
        cudaMemcpy(maskPixels,maskBuffer,nHeight*nWidth*sizeof(int32_t),cudaMemcpyDeviceToHost);
	}
}

__host__ void CopyCenterListDeviceToHost(SLICClusterCenter* centerList)
{
	if (cudaIsInitialized)
	{
		cudaMemcpy(centerList,vSLICCenterList,nMaxSegs*sizeof(SLICClusterCenter),cudaMemcpyDeviceToHost);
	}
}

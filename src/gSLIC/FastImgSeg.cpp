
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <time.h>
#include <iostream>
#include <fstream>

#include "FastImgSeg.h"
#include "cudaSegEngine.h"


using namespace std;

FastImgSeg::FastImgSeg(int w, int h, int d, int nSegments)
{
    sourceImage     = NULL;
    segMask         = NULL;
    markedImg       = NULL;
    centerList      = NULL;

    initializeFastSeg(w, h, nSegments);
}

FastImgSeg::FastImgSeg()
{
    sourceImage     = NULL;
    segMask         = NULL;
    markedImg       = NULL;
    centerList      = NULL;
}

FastImgSeg::~FastImgSeg()
{
	clearFastSeg();
}

void FastImgSeg::changeClusterNum(int nSegments)
{
    nSeg = nSegments;
}

void FastImgSeg::initializeFastSeg(int w, int h, int nSegments)
{
    if( segMask != NULL ) clearFastSeg();

	int nAvailSegs=(iDivUp(w,BLCK_SIZE))*(iDivUp(h,BLCK_SIZE));
	if(nAvailSegs < nSegments) {
        //printf("Max no. of segment is %d\n", nAvailSegs);
		nSegments = nAvailSegs;
	}

	// MaxSegs should be same on InitCUDA()@CudaSegEngine.cu
    nMaxSegs = (iDivUp(w,BLCK_SIZE)*2)*(iDivUp(h,BLCK_SIZE)*2);

    width       = w;
    height      = h;
    nSeg        = nSegments;

    segMask     = (int32_t*) malloc(width*height*sizeof(int32_t));
    markedImg   = (unsigned char*)malloc(width*height*4*sizeof(unsigned char));

    centerList  = (SLICClusterCenter*)malloc(nMaxSegs*sizeof(SLICClusterCenter));

    InitCUDA(width, height, nSegments, SLIC);

    bImgLoaded  = false;
    bSegmented  = false;
}

void FastImgSeg::clearFastSeg()
{
	free(segMask);
	free(markedImg);
	free(centerList);

	TerminateCUDA();

    bImgLoaded = false;
    bSegmented = false;
}

void FastImgSeg::LoadImg(unsigned char* imgP)
{
	sourceImage=imgP;
	CUDALoadImg(sourceImage);
	memcpy(markedImg,sourceImage,width*height*4*sizeof(unsigned char));
	bSegmented=false;
}

void FastImgSeg::DoSegmentation(SEGMETHOD eMethod, double weight)
{
	clock_t start,finish;

	start=clock();
	CudaSegmentation(eMethod, weight);
	finish=clock();
    printf("clustering:%f \n",(double)(finish-start)/CLOCKS_PER_SEC);

	CopyMaskDeviceToHost(segMask);
	CopyCenterListDeviceToHost(centerList);

    if( 1 ) {
        start=clock();
        enforceConnectivity(segMask,width,height,nMaxSegs);
        finish=clock();
        printf("connectivity:%f\n",(double)(finish-start)/CLOCKS_PER_SEC);
    }

	bSegmented=true;
}


void FastImgSeg::DoSegmentation(uint8_t *imgBuf, int32_t *segBuf, SEGMETHOD eMethod, double weight)
{
    bSegmented = false;

    // load image
    sourceImage=imgBuf;
    CUDALoadImg(sourceImage);
    memcpy(markedImg, sourceImage, width*height*4*sizeof(unsigned char));

    int32_t *segMaskOld = segMask;
    segMask = segBuf;

    CudaSegmentation(eMethod, weight);

    CopyMaskDeviceToHost(segMask);
    CopyCenterListDeviceToHost(centerList);

    if( 1 ) {
        enforceConnectivity(segMask, width, height, nMaxSegs);
    }

    segMask = segMaskOld;
    bSegmented = true;
}

void FastImgSeg::Tool_GetMarkedImg()
{
	if (!bSegmented)
		return;

	for (int i=1;i<height-1;i++) {
		for (int j=1;j<width-1;j++) {
			int mskIndex=i*width+j;
			if (segMask[mskIndex]!=segMask[mskIndex+1] 
			|| segMask[mskIndex]!=segMask[(i-1)*width+j]
			|| segMask[mskIndex]!=segMask[mskIndex-1]
			|| segMask[mskIndex]!=segMask[(i+1)*width+j])
			{
				markedImg[mskIndex*4]=0;
				markedImg[mskIndex*4+1]=0;
				markedImg[mskIndex*4+2]=255;
			}
		}
	}
}

void FastImgSeg::Tool_GetFilledImg()
{
	if (!bSegmented)
		return;

	// Fill with lab color
	for (int i=0;i<height;i++) {
		for (int j=0;j<width;j++) {
			int mskIndex=i*width+j;
			int centerIndex = segMask[mskIndex];

			if(centerIndex >= nMaxSegs) {
				printf("[%s:%d] centerIndex(%d) is greater than or equals nMaxSegs(%d)\n", __FILE__, __LINE__, centerIndex, nMaxSegs);
				continue;
			}

			float4 lab = centerList[centerIndex].lab;
			float x = lab.x * 255;
			float y = lab.y * 255;
			float z = lab.z * 255;

			int r = 3.24071 * x + (-1.53726) * y + (-0.498571) * z;
			int g =(-0.969258) * x + 1.87599 * y + 0.0415557 * z;
			int b = 0.0556352 * x + (-0.203996) * y + 1.05707 * z;

			markedImg[mskIndex*4 + 0] = b;
			markedImg[mskIndex*4 + 1] = g;
			markedImg[mskIndex*4 + 2] = r;
		}
	}
}

void FastImgSeg::Tool_DrawSites()
{
	if (!bSegmented)
		return;

	// DRAW Center
	for(int i = 0; i < nMaxSegs; i ++) {
		float2 srcXY = centerList[i].xy;
		int srcX = srcXY.x;
		int srcY = srcXY.y;

		if(0 <= srcX && srcX < width && 0 <= srcY && srcY < height) {
			int srcIndex = srcY*width+srcX;

			markedImg[srcIndex*4 + 0] = 255;
			markedImg[srcIndex*4 + 1] = 255;
			markedImg[srcIndex*4 + 2] = 255;
		}
	}
}

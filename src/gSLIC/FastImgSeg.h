#ifndef __CUDA_SUPERPIXELSEG__
#define __CUDA_SUPERPIXELSEG__

#include <stdint.h>

#include "cudaUtil.h"
#include "cudaSegSLIC.h"

class FastImgSeg
{
public:
	FastImgSeg();
    FastImgSeg(int width, int height, int dim, int nSegments);
	~FastImgSeg();

    void initializeFastSeg(int width, int height, int nSegments);
	void clearFastSeg();
	void changeClusterNum(int nSegments);

	void LoadImg(unsigned char* imgP);
	void DoSegmentation(SEGMETHOD eMethod, double weight);

    void DoSegmentation(uint8_t *imgBuf, int32_t *segBuf, SEGMETHOD eMethod, double weight);

	void Tool_GetMarkedImg();
	void Tool_GetFilledImg();
	void Tool_DrawSites();

public:
    unsigned char*      sourceImage;
    unsigned char*      markedImg;
    int32_t*            segMask;
    SLICClusterCenter*  centerList;
    int                 nMaxSegs;

private:
    int                 width;
    int                 height;
    int                 nSeg;

    bool                bSegmented;
    bool                bImgLoaded;
    bool                bMaskGot;
};

#endif

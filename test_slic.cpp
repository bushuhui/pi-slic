#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include <set>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/legacy/legacy.hpp>


#ifdef ENABLE_GPU
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <helper_functions.h>       // helper for shared functions common to CUDA SDK samples
#include <helper_cuda.h>            // helper functions for CUDA error checking and initialization
#include <helper_cuda_drvapi.h>     // helper functions for drivers
#endif

#include <base/time/Time.h>
#include <base/debug/debug_config.h>
#include <base/utils/utils.h>

#include "src/vl/slic.h"
#include "src/gSLIC/FastImgSeg.h"
#include "src/PI_SLIC.h"

using namespace std;
using namespace pi;



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int test_cslic(CParamArray *pa)
{
    int testNum = svar.GetInt("testNum", 1);
    int useGPU = svar.GetInt("useGPU", 0);

    // load image    
    string imgFN = svar.GetString("image", "./data/test640.png");

    cv::Mat img = cv::imread(imgFN);

    int imgW, imgH, imgC, block;
    imgW  = img.cols;
    imgH  = img.rows;
    imgC  = img.channels();
    block = imgW * imgH;

    dbg_pt("input img W, H, C = %3d, %3d, %3d\n", imgW, imgH, imgC);

    cv::Mat img2;
    uint32_t *seg;
    float *imgF;

    if( 1 ) {
        if( imgC == 3 )
            cv::cvtColor(img, img2, CV_BGR2GRAY);
        else
            img2 = img;

        // do SLIC segmentation
        imgF = new float[imgW*imgH];
        seg = new uint32_t[imgW*imgH];

        uint8_t *pix = img2.data;
        for(int i=0; i<imgW*imgH; i++) imgF[i] = pix[i];
    } else {
        if( imgC == 1 )
            cv::cvtColor(img, img2, CV_GRAY2BGR);
        else
            img2 = img;

        // do SLIC segmentation
        imgF = new float[imgW*imgH*3];
        seg = new uint32_t[imgW*imgH];

        uint8_t *pix = img2.data;
        for(int iy=0; iy<imgH; iy++) {
            for(int ix=0; ix<imgW; ix++) {
                for(int ic=0; ic<3; ic++) {
                    imgF[ic*imgW*imgH + iy*imgW + ix] = *pix;
                    pix++;
                }
            }
        }
    }

    int regionSize = 10;
    float regularization = 0.1;
    int minRegionSize = 16;
    regionSize      = svar.GetInt("regionSize", 10);
    regularization  = svar.GetDouble("regularization", 0.1);
    minRegionSize   = svar.GetInt("minRegionSize", 16);


    for(int i=0; i<testNum; i++) {
        timer.enter("vl_slic_segment");
        vl_slic_segment(seg, imgF,
                        imgW, imgH, 1,
                        regionSize, regularization, minRegionSize);
        timer.leave("vl_slic_segment");
    }

    // draw segmentation results
    uint32_t segIMax = 0;
    set<uint32_t> segSet;
    for(int i=0; i<imgW*imgH; i++) {
        uint32_t si = seg[i];

        if( si > segIMax ) segIMax = si;

        set<uint32_t>::iterator it = segSet.find(si);
        if( it == segSet.end() ) segSet.insert(si);
    }

    int segN = segSet.size();
    printf("segNum = %d, segIMax = %d\n", segN, segIMax);

    uint8_t *colorLUT = new uint8_t[segIMax*3];
    for(int i=0; i<segIMax; i++) {
        colorLUT[i*3 + 0] = rand() % 256;
        colorLUT[i*3 + 1] = rand() % 256;
        colorLUT[i*3 + 2] = rand() % 256;
    }

    cv::Mat imgSeg;
    imgSeg.create(imgH, imgW, CV_8UC3);
    uint8_t *pImg = imgSeg.data;
    for(int i=0; i<imgW*imgH; i++) {
        uint32_t si = seg[i];

        pImg[i*3 + 0] = colorLUT[si*3 + 0];
        pImg[i*3 + 1] = colorLUT[si*3 + 1];
        pImg[i*3 + 2] = colorLUT[si*3 + 2];
    }

    string fnImgOut = fmt::sprintf("%s_out.png", imgFN);
    cv::imwrite(fnImgOut, imgSeg);


    delete [] imgF;
    delete [] seg;
    delete [] colorLUT;

    return 0;
}


// Copy cvImage to memory buffer
void CvImgToBuffer(IplImage* frame, unsigned char* imgBuffer)
{
    for (int i=0;i<frame->height;i++) {
        for (int j=0;j<frame->width;j++) {
            int bufIdx=(i*frame->width+j)*4;

            imgBuffer[bufIdx]=CV_IMAGE_ELEM(frame,unsigned char,i,j*3);
            imgBuffer[bufIdx+1]=CV_IMAGE_ELEM(frame,unsigned char,i,j*3+1);
            imgBuffer[bufIdx+2]=CV_IMAGE_ELEM(frame,unsigned char,i,j*3+2);
        }
    }
}

// Copy memory buffer to cvImage
void OutputImgToCvImg(unsigned char* markedImg, IplImage* frame)
{
    for (int i=0;i<frame->height;i++) {
        for (int j=0;j<frame->width;j++) {
            int bufIdx=(i*frame->width+j)*4;
            CV_IMAGE_ELEM(frame,unsigned char,i,j*3)=markedImg[bufIdx];
            CV_IMAGE_ELEM(frame,unsigned char,i,j*3+1)=markedImg[bufIdx+1];
            CV_IMAGE_ELEM(frame,unsigned char,i,j*3+2)=markedImg[bufIdx+2];
        }
    }
}

CvSubdiv2D* InitSubdivision(CvMemStorage* storage, CvRect rect)
{
    CvSubdiv2D* subdiv;

    subdiv = cvCreateSubdiv2D( CV_SEQ_KIND_SUBDIV2D, sizeof(*subdiv), sizeof(CvSubdiv2DPoint), sizeof(CvQuadEdge2D), storage);

    cvInitSubdivDelaunay2D(subdiv, rect);

    return subdiv;
}

void LocatePoint(CvSubdiv2D* subdiv, CvPoint2D32f fp)
{
    CvSubdiv2DEdge e;
    CvSubdiv2DEdge e0 = 0;
    CvSubdiv2DPoint* p = 0;

    cvSubdiv2DLocate(subdiv, fp, &e0, &p);

    if(e0) {
        e = e0;

        do {
            e = cvSubdiv2DGetEdge(e, CV_NEXT_AROUND_LEFT);
        }
        while( e != e0 );
    }
}

CvSeq* ExtractSeq(CvSubdiv2DEdge edge, CvMemStorage* storage)
{
    CvSeq* pSeq = NULL;

    CvSubdiv2DEdge egTemp = edge;
    int i, nCount = 0;
    CvPoint* buf = 0;

    // count number of edges in facet
    do {
        nCount ++;
        egTemp = cvSubdiv2DGetEdge(egTemp, CV_NEXT_AROUND_LEFT );
    }
    while(egTemp != edge);

    buf = (CvPoint*)malloc(nCount * sizeof(buf[0]));

    // gather points
    egTemp = edge;

    for( i = 0; i < nCount; i++ ) {
        CvSubdiv2DPoint* pt = cvSubdiv2DEdgeOrg(egTemp);

        if(!pt)
            break;

        CvPoint ptInsert = cvPoint( cvRound(pt->pt.x), cvRound(pt->pt.y));

        buf[i] = ptInsert;

        egTemp = cvSubdiv2DGetEdge(egTemp, CV_NEXT_AROUND_LEFT );
    }

    if(i == nCount) {
        pSeq = cvCreateSeq(CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvContour), sizeof(CvPoint), storage);

        for( i = 0; i < nCount; i ++)
            cvSeqPush(pSeq, &buf[i]);
    }

    free( buf );

    return pSeq;
}

void DrawVoronoiDiagram(IplImage* image, SLICClusterCenter* centerList, int listSize)
{
    CvRect cvRect = {0, 0, image->width, image->height};

    CvMemStorage* pStorage = cvCreateMemStorage(0);

    // Init Subdivision
    CvSubdiv2D* pSubDiv = InitSubdivision(pStorage, cvRect);

    // Add sites
    for(int i = 0; i < listSize; i ++) {
        SLICClusterCenter center = centerList[i];

        float x = center.xy.x;
        float y = center.xy.y;

        if(0 <= x && x < image->width && 0 <= y && y < image->height) {
            CvPoint2D32f fPoint = cvPoint2D32f(x, y);
            LocatePoint(pSubDiv, fPoint);
            cvSubdivDelaunay2DInsert(pSubDiv, fPoint);
        }
    }

    // Calculate voronoi tessellation
    cvCalcSubdivVoronoi2D(pSubDiv);

    // Draw edges
    int nEdgeCount = pSubDiv->edges->total;
    int nElementSize = pSubDiv->edges->elem_size;

    CvPoint** ppPoints = new CvPoint*[1];
    ppPoints[0] = new CvPoint[2048];
    int pnPointCount[1];

    CvSeqReader reader;

    cvStartReadSeq( (CvSeq*)(pSubDiv->edges), &reader, 0);

    for(int i = 0; i < nEdgeCount; i++) {
        CvQuadEdge2D* edge = (CvQuadEdge2D*)(reader.ptr);

        if( CV_IS_SET_ELEM( edge )) {
            CvSubdiv2DEdge e = (CvSubdiv2DEdge)edge;

            CvSeq *pSeq = ExtractSeq(cvSubdiv2DRotateEdge( e, 1 ), pStorage);

            if(pSeq != NULL) {
                pnPointCount[0] = pSeq->total;

                for(int j = 0; j < pSeq->total; j ++) {
                    CvPoint pt = *CV_GET_SEQ_ELEM(CvPoint, pSeq, j);
                    ppPoints[0][j] = cvPoint(pt.x, pt.y);
                }

                cvPolyLine(image, ppPoints, pnPointCount, 1, -1, CV_RGB(0, 0, 255));
            }
        }

        CV_NEXT_SEQ_ELEM( nElementSize, reader);
    }

    delete [] ppPoints[0];
    delete [] ppPoints;

    cvReleaseMemStorage(&pStorage);
}

int test_gslic_1(pi::CParamArray *pa)
{
    IplImage* frame = cvLoadImage("./data/photo.png");

    FastImgSeg* mySeg = new FastImgSeg();
    mySeg->initializeFastSeg(frame->width,frame->height, 2000);

    unsigned char* imgBuffer=(unsigned char*)malloc(frame->width * frame->height * sizeof(unsigned char) * 4);

    CvImgToBuffer(frame, imgBuffer);

    mySeg->LoadImg(imgBuffer);

    cvNamedWindow("frame",0);

    float weight = 5.1f;
    mySeg->DoSegmentation(SLIC, weight);

    mySeg->Tool_GetFilledImg();
    //mySeg->Tool_GetMarkedImg();
    //mySeg->Tool_DrawSites();
    OutputImgToCvImg(mySeg->markedImg, frame);

    cvShowImage("frame",frame);

    //mySeg->centerList
    DrawVoronoiDiagram(frame, mySeg->centerList, mySeg->nMaxSegs);

    cvShowImage("frame",frame);

    cvWaitKey(0);

    cvSaveImage("photo_segmented.png", frame);

    cvDestroyWindow( "frame" );

    cvFree(&frame);
    delete mySeg;

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int test_gslic(CParamArray *pa)
{
    int testNum = svar.GetInt("testNum", 1);
    int useGPU = svar.GetInt("useGPU", 0);

    // load image
    string imgFN = svar.GetString("image", "./data/test640.png");

    cv::Mat img = cv::imread(imgFN);

    int imgW, imgH, imgC, block;
    imgW  = img.cols;
    imgH  = img.rows;
    imgC  = img.channels();
    block = imgW * imgH;

    dbg_pt("input img W, H, C = %3d, %3d, %3d\n", imgW, imgH, imgC);

    cv::Mat img2;
    int *seg;
    uint8_t *imgBuf;

    {
        if( imgC == 1 )
            cv::cvtColor(img, img2, CV_GRAY2BGR);
        else
            img2 = img;

        // convert image
        imgBuf = new uint8_t[imgW*imgH*4];

        uint8_t *pix = img2.data;
        for(int iy=0; iy<imgH; iy++) {
            for(int ix=0; ix<imgW; ix++) {
                imgBuf[(iy*imgW + ix)*4 + 0] = *(pix++);
                imgBuf[(iy*imgW + ix)*4 + 1] = *(pix++);
                imgBuf[(iy*imgW + ix)*4 + 2] = *(pix++);
            }
        }
    }

    int segNum = 2000;
    float weight = 0.1;
    segNum  = svar.GetInt("segNum", 2000);
    weight  = svar.GetDouble("weight", 0.1);

    FastImgSeg* mySeg = new FastImgSeg();
    mySeg->initializeFastSeg(imgW, imgH, segNum);
    mySeg->LoadImg(imgBuf);

    for(int i=0; i<testNum; i++) {
        timer.enter("gSLIC");
        mySeg->DoSegmentation(SLIC, weight);
        timer.leave("gSLIC");
    }

    // draw segmentation results
    seg = mySeg->segMask;
    int segIMax = 0;
    set<int> segSet;
    for(int i=0; i<imgW*imgH; i++) {
        int si = seg[i];

        if( si > segIMax ) segIMax = si;

        set<int>::iterator it = segSet.find(si);
        if( it == segSet.end() ) segSet.insert(si);
    }

    int segN = segSet.size();
    printf("segNum = %d, segIMax = %d\n", segN, segIMax);

    uint8_t *colorLUT = new uint8_t[segIMax*3];
    for(int i=0; i<segIMax; i++) {
        colorLUT[i*3 + 0] = rand() % 256;
        colorLUT[i*3 + 1] = rand() % 256;
        colorLUT[i*3 + 2] = rand() % 256;
    }

    cv::Mat imgSeg;
    imgSeg.create(imgH, imgW, CV_8UC3);
    uint8_t *pImg = imgSeg.data;
    for(int i=0; i<imgW*imgH; i++) {
        int si = seg[i];

        pImg[i*3 + 0] = colorLUT[si*3 + 0];
        pImg[i*3 + 1] = colorLUT[si*3 + 1];
        pImg[i*3 + 2] = colorLUT[si*3 + 2];
    }

    string fnImgOut = fmt::sprintf("%s_out.png", imgFN);
    cv::imwrite(fnImgOut, imgSeg);


    delete [] imgBuf;
    delete [] colorLUT;

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int test_slic(CParamArray *pa)
{
    int testNum = svar.GetInt("testNum", 1);
    int useGPU = svar.GetInt("useGPU", 0);

    // load image
    string imgFN = svar.GetString("image", "./data/photo.png");

    cv::Mat img = cv::imread(imgFN);

    int imgW, imgH, imgC, block;
    imgW  = img.cols;
    imgH  = img.rows;
    imgC  = img.channels();
    block = imgW * imgH;

    dbg_pt("input img: %s", imgFN.c_str());
    dbg_pt("input img W, H, C = %3d, %3d, %3d\n", imgW, imgH, imgC);



    // load parameters
    int   regionSize      = svar.GetInt("regionSize", 16);
    float regularization  = svar.GetDouble("regularization", 0.1);
    int   minRegionSize   = svar.GetInt("minRegionSize", 16);

    // do segmentation
    PI_SLIC slic;

    slic.setMode(imgW, imgH, useGPU, regionSize, regularization, minRegionSize);

    int32_t *segBuf = new int32_t[imgW*imgH];

    for(int i=0; i<testNum; i++) {
        timer.enter("PI_SLIC");
        slic.doSegment(img, segBuf);
        timer.leave("PI_SLIC");
    }


    // draw segmentation results
    int segIMax = 0;
    set<int32_t> segSet;
    for(int i=0; i<imgW*imgH; i++) {
        int32_t si = segBuf[i];

        if( si > segIMax ) segIMax = si;

        set<int32_t>::iterator it = segSet.find(si);
        if( it == segSet.end() ) segSet.insert(si);
    }

    int segN = segSet.size();
    dbg_pt("segNum = %d, segIMax = %d\n", segN, segIMax);

    uint8_t *colorLUT = new uint8_t[segIMax*3];
    for(int i=0; i<segIMax; i++) {
        colorLUT[i*3 + 0] = rand() % 256;
        colorLUT[i*3 + 1] = rand() % 256;
        colorLUT[i*3 + 2] = rand() % 256;
    }

    cv::Mat imgSeg;
    imgSeg.create(imgH, imgW, CV_8UC3);
    uint8_t *pImg = imgSeg.data;
    for(int i=0; i<imgW*imgH; i++) {
        int si = segBuf[i];

        pImg[i*3 + 0] = colorLUT[si*3 + 0];
        pImg[i*3 + 1] = colorLUT[si*3 + 1];
        pImg[i*3 + 2] = colorLUT[si*3 + 2];
    }

    string fnImgOut = fmt::sprintf("%s_out_%d.png", imgFN, useGPU);
    cv::imwrite(fnImgOut, imgSeg);


    delete [] segBuf;
    delete [] colorLUT;

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

struct TestFunctionArray g_fa[] =
{
    TEST_FUNC_DEF(test_cslic,                   "Test SLIC segmentation (CPU)"),

    TEST_FUNC_DEF(test_gslic_1,                 "Test gSLIC segmentation (demo 1)"),
    TEST_FUNC_DEF(test_gslic,                   "Test gSLIC segmentation"),

    TEST_FUNC_DEF(test_slic,                    "Test SLIC segmentation"),


    {NULL,  "NULL",  "NULL"},
};


int main(int argc, char *argv[])
{
    // setup debug trace
    dbg_stacktrace_setup();

    // run function
    return svar_main(argc, argv, g_fa);
}



#include <opencv2/imgproc/imgproc.hpp>

#include <base/debug/debug_config.h>


#include "PI_SLIC.h"

#include "vl/slic.h"
#ifdef ENABLE_GPU
#include "gSLIC/FastImgSeg.h"
#endif


PI_SLIC::PI_SLIC()
{
    m_imgW = 0;
    m_imgH = 0;

    m_useGPU = 0;

    m_regionSize    = 16;
    m_regulation    = 0.1;
    m_minRegionSize = 16;
    m_colorType     = COLORTYPE_LAB;

#ifdef ENABLE_GPU
    FastImgSeg *is = new FastImgSeg;
    m_data = is;
#else
    m_data = NULL;
#endif
}

PI_SLIC::~PI_SLIC()
{
#ifdef ENABLE_GPU
    FastImgSeg *is = (FastImgSeg*) m_data;
    if( is != NULL ) {
        delete is;
        m_data = NULL;
    }
#endif
}

int PI_SLIC::setMode(int imgW, int imgH,
            int useGPU,
            int regionSize, float regulation, int minRegionSize,
            SLIC_ColorType ct)
{
    m_imgW          = imgW;
    m_imgH          = imgH;

    m_regionSize    = regionSize;
    m_regulation    = regulation;
    m_minRegionSize = minRegionSize;
    m_colorType     = ct;

    m_segmentNum    = (int)( ceil(m_imgW*1.0/m_regionSize)*ceil(m_imgH*1.0/m_regionSize) );


    if( useGPU ) {
#ifdef ENABLE_GPU
        FastImgSeg *fis = (FastImgSeg*) m_data;
        fis->initializeFastSeg(m_imgW, m_imgH, m_segmentNum);

        m_useGPU = 1;
#else
        dbg_pe("Please compile the code with CUDA by define ENABLE_GPU!");
        m_useGPU = 0;
#endif
    }

    return 0;
}


int PI_SLIC::doSegment(const cv::Mat &img, int32_t *segment)
{
    int imgC = img.channels();
    int imgW = img.cols, imgH = img.rows;

    cv::Mat imgIn;

    if( m_useGPU ) {
#ifdef ENABLE_GPU
        FastImgSeg *fis = (FastImgSeg*) m_data;

        if( imgW != m_imgW || imgH != m_imgH ) {
            setMode(imgW, imgH, m_useGPU, m_regionSize, m_regulation, m_minRegionSize, m_colorType);
        }

        // create 4-channel image
        imgIn.create(imgH, imgW, CV_8UC4);
        if( img.channels() == 3 )
            cv::cvtColor(img, imgIn, CV_BGR2BGRA);
        else
            cv::cvtColor(img, imgIn, CV_GRAY2BGRA);

        SEGMETHOD sm = SLIC;
        if( m_colorType == COLORTYPE_RGB ) sm = RGB_SLIC;

        fis->DoSegmentation(imgIn.data, segment, sm, m_regulation);

#else
        dbg_pe("Please compile the code with CUDA by define ENABLE_GPU!");
        return 0;
#endif
    } else {
        if( imgC == 3 )
            cv::cvtColor(img, imgIn, CV_BGR2GRAY);
        else
            imgIn = img;

        // do SLIC segmentation
        float *imgF = new float[imgW*imgH];
        uint8_t *pix = imgIn.data;

        for(int i=0; i<imgW*imgH; i++) imgF[i] = pix[i];

        vl_slic_segment((vl_uint32*) segment, imgF,
                        imgW, imgH, 1,
                        m_regionSize, m_regulation, m_minRegionSize);

        delete [] imgF;
    }

    return 0;
}

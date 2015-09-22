#ifndef __PI_SLIC_H__
#define __PI_SLIC_H__

#include <stdint.h>
#include <opencv2/core/core.hpp>


class PI_SLIC
{
public:
    enum SLIC_ColorType {
        COLORTYPE_GRAY,
        COLORTYPE_RGB,
        COLORTYPE_LAB
    };

public:
    PI_SLIC();
    virtual ~PI_SLIC();

    int setMode(int imgW, int imgH,
                int useGPU = 0,
                int regionSize = 16, float regulation = 0.1, int minRegionSize = 16,
                SLIC_ColorType ct = COLORTYPE_LAB);

    int doSegment(const cv::Mat &img, int32_t *segment);

protected:
    int                 m_useGPU;                   ///< use GPU or not

    int                 m_imgW, m_imgH;             ///< input image size
    int                 m_regionSize;               ///< region size (in x,y direction pixel number)
    int                 m_segmentNum;               ///< segment number
    float               m_regulation;               ///< regulation coefficient
    int                 m_minRegionSize;            ///< mini-region size
    SLIC_ColorType      m_colorType;                ///< color type

    void                *m_data;
};


#endif // end of __PI_SLIC_H__

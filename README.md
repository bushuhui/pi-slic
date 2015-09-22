# PI-SLIC - high-performance superpixel segmentation toolbox for C++

This program is a C++ toolbox for generating superpixel from image. For achieving high-performance computation, it support CUDA acceleration. The average time for a 640x480 image is 40 ms. You can easly integrated the code to you embedded program. The core functions are extracted from vlfeat (http://www.vlfeat.org/) and gSLIC (https://github.com/painnick/gSLIC). We fix some problems of their code and integrated into a package.



## Requirements:
* OpenCV 2.4.9 (or above)
* CUDA 5.0 (or above)
* PIL (included in the code at ./Thirdparty/PIL)

## Compile:

*1. build PIL*
```
cd ./Thirdparty/PIL
make
```

*3. build pi-slic*
```
make
```



## Usage:

```
# do superpixel with GPU
./test_slic useGPU=1

# do superpixel without GPU
./test_slic useGPU=1
```

## Plateform:
Only test on LinuxMint 17.1 64-bit, may be other distributions are also support. 


## Screenshot:
-![alt text](http://www.adv-ci.com/blog/wp-content/uploads/2015/09/screenshot_2-275x300.png "Screenshot 1")
-![alt text](http://www.adv-ci.com/blog/wp-content/uploads/2015/09/screenshot_1-1024x559.png "Screenshot 2")


## Project homepage:
http://www.adv-ci.com/blog/source/pi-slic/


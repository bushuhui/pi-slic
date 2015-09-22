###############################################################################
###############################################################################
CC                  = gcc
CXX                 = g++

ENABLE_GPU         ?= true

###############################################################################
###############################################################################
DEFAULT_CFLAGS      = -D_GNU_SOURCE -O3 -DNDEBUG -fPIC
DEFAULT_CFLAGS     += -g -rdynamic
#DEFAULT_LDFLAGS     = -lstdc++ -lpthread 

#DEFAULT_CFLAGS     += -fopenmp -pthread
#DEFAULT_LDFLAGS    += -fopenmp


################################################################################
# OpenCV settings
# run following command first:
#   export PKG_CONFIG_PATH=/opt/opencv-2.4/lib/pkgconfig
################################################################################
OPENCV_CFLAGS       = $(shell pkg-config --cflags opencv)
OPENCV_LDFLAGS      = $(shell pkg-config --libs   opencv) 


###############################################################################
###############################################################################
CUDA_DIR            = /usr/local/cuda
CUDA_NVCC           = $(CUDA_DIR)/bin/nvcc
CUDA_CFLAGS         = -I$(CUDA_DIR)/include 
CUDA_LDFLAGS        = -L$(CUDA_DIR)/lib64 -lcublas -lcudart


###############################################################################
###############################################################################
PIL_DIR             = ./Thirdparty/PIL
PIL_CFLAGS          = -I$(PIL_DIR)/src -DPIL_LINUX 
PIL_LDFLAGS         = -L$(PIL_DIR)/libs -lpi_base \
                      -Wl,-rpath=$(PIL_DIR)/libs

###############################################################################
###############################################################################
CUDAUTILS_DIR       = ./Thirdparty/cuda_utils_5.0
CUDAUTILS_CFLAGS    = -I$(CUDAUTILS_DIR)/inc
#CUDAUTILS_LDFLAGS   = -L$(CUDAUTILS_DIR)/lib -lcuda_utils


###############################################################################
###############################################################################
LIBS_CFLAGS         = $(PIL_CFLAGS)  $(OPENCV_CFLAGS)
LIBS_LDFLAGS        = $(PIL_LDFLAGS) $(OPENCV_LDFLAGS)

CFLAGS              = $(DEFAULT_CFLAGS)  $(LIBS_CFLAGS)
LDFLAGS             = $(DEFAULT_LDFLAGS) $(LIBS_LDFLAGS)


ifneq ($(ENABLE_GPU),)

LIBS_CFLAGS        += $(CUDAUTILS_CFLAGS)
LIBS_LDFLAGS       += $(CUDAUTILS_LDFLAGS)

CFLAGS             +=  $(CUDA_CFLAGS) $(CUDAUTILS_CFLAGS) \
                        -DENABLE_GPU 
LDFLAGS            += $(CUDA_LDFLAGS) $(CUDAUTILS_LDFLAGS)


CUFLAGS             = -ccbin $(CXX) \
                        --compiler-options=-fPIC \
                        $(LIBS_CFLAGS) \
                        -DENABLE_GPU 

CUFLAGS			   += -arch=sm_21 
endif


###############################################################################
###############################################################################

cSLIC_src           = src/vl/mathop.c src/vl/generic.c src/vl/host.c src/vl/mathop_sse2.c \
                      src/vl/random.c src/vl/slic.c 
cSLIC_src          += src/PI_SLIC.cpp

ifeq ($(ENABLE_GPU),)
#gSLIC_src           = gSLIC/cudaImgTrans.cu gSLIC/cudaSegEngine.cu gSLIC/cudaSegSLIC.cu gSLIC/cudaUtil.cu \
#                      gSLIC/FastImgSeg.cpp  
else
gSLIC_src           = src/gSLIC/cudaImgTrans.cu src/gSLIC/cudaSegEngine.cu \
                      src/gSLIC/cudaSegSLIC.cu src/gSLIC/cudaUtil.cu \
                      src/gSLIC/FastImgSeg.cpp  
endif

cSLIC_tgt          := $(patsubst %.cpp,%.o,$(cSLIC_src))
cSLIC_tgt          := $(patsubst %.c,%.o,$(cSLIC_tgt))

gSLIC_obj          := $(patsubst %.cpp,%.o,$(gSLIC_src))
gSLIC_obj          := $(patsubst %.cu,%.o,$(gSLIC_obj))

target              = test_slic libpi_slic.so

###############################################################################
###############################################################################

all : $(cpp_tgt) $(target)


%.o : %.cpp
	$(CXX) -c $? -o $(@) $(CFLAGS) 

%.o : %.c
	$(CC) -c $? -o $(@) $(CFLAGS) 


# CUDA codes
ifeq ($(ENABLE_GPU),)

libpi_slic.so : $(cSLIC_tgt) 
	$(CXX) -o $@ $(cSLIC_tgt) -shared $(LDFLAGS)

else

%.o : %.cu
	$(CUDA_NVCC) -c $? -o $(@) $(CUFLAGS)

libpi_slic.so : $(cSLIC_tgt) $(gSLIC_obj)
	$(CXX) -o $@ $(cSLIC_tgt) $(gSLIC_obj) -shared $(LDFLAGS)

endif


test_slic : test_slic.cpp libpi_slic.so 
	$(CXX) $< -o $(@) $(CFLAGS) $(LDFLAGS) -L. -lpi_slic -Wl,-rpath=.


clean:
	rm -f $(target) src/*.o src/vl/*.o src/gSLIC/*.o


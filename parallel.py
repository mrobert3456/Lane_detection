import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda import driver
import numpy
import os

import cv2 as cv
import math

BLOCK_SIZE = 32

if (os.system("cl.exe")):
    os.environ['PATH'] += ';' + r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.28.29910\bin\Hostx64\x64"


def GrayScaleGPU(img):
    result = numpy.empty_like(img)
    R = img[:, :, 0].copy()
    G = img[:, :, 1].copy()
    B = img[:, :, 2].copy()

    height = img.shape[0]
    width = img.shape[1]

    dim_gridx = math.ceil(width / BLOCK_SIZE)
    dim_gridy = math.ceil(height / BLOCK_SIZE)

    dev_R = cuda.mem_alloc(R.nbytes)
    dev_G = cuda.mem_alloc(G.nbytes)
    dev_B = cuda.mem_alloc(B.nbytes)

    # copy to gpu
    cuda.memcpy_htod(dev_R, R)
    cuda.memcpy_htod(dev_G, G)
    cuda.memcpy_htod(dev_B, B)

    mod = SourceModule("""
            __global__ void Convert_To_Gray(unsigned char * R, unsigned char * G, unsigned char * B, const unsigned int width, const unsigned int height)
            {
                const unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
                const unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;

                if (row <height && col<width)
                {
                    const unsigned int idx = col+row*width;
                    const unsigned char intensity = R[idx]*0.07+G[idx]*0.72+B[idx]*0.21;

                    R[idx] = intensity;
                    G[idx] = intensity;
                    B[idx] = intensity;
                 }
                else{
                return;
                }

        }

    """)

    grayConv = mod.get_function("Convert_To_Gray")

    block_count = (height * width - 1) / BLOCK_SIZE * BLOCK_SIZE + 1
    grayConv(dev_R,
             dev_G,
             dev_B,
             numpy.uint32(width),
             numpy.uint32(height),
             block=(BLOCK_SIZE, BLOCK_SIZE, 1),  # 1 blokk 32*32 -es-> tehát 1024 db szal van benne
             grid=(dim_gridx, dim_gridy)  # blokkok osszessege
             )

    # fullhd.jpg az 900*1600 as, ami = 1440000 pixel
    # 1440000/1024 = 1407 kerekitve -> ez a block_count

    # copy result from gpu
    R_new = numpy.empty_like(R)
    cuda.memcpy_dtoh(R_new, dev_R)

    G_new = numpy.empty_like(G)
    cuda.memcpy_dtoh(G_new, dev_G)

    B_new = numpy.empty_like(B)
    cuda.memcpy_dtoh(B_new, dev_B)

    result[:, :, 0] = R_new
    result[:, :, 1] = G_new
    result[:, :, 2] = B_new

    cannyimg = cv.Canny(result, 100, 200)  # use canny edge detection on the grayscale img
    return cannyimg


def HistogramGPU(img):
    res = img[0, :]
    img_uj = img[int(img.shape[0] / 2):, :]
    height = img.shape[0] / 2
    width = img.shape[1]
    dim_gridx = math.ceil(width / BLOCK_SIZE)
    dim_gridy = math.ceil(height / BLOCK_SIZE)




    dev_Hist = cuda.mem_alloc(res.nbytes)
    dev_kep = cuda.mem_alloc(img_uj.nbytes)

    cuda.memcpy_htod(dev_Hist, res)
    cuda.memcpy_htod(dev_kep, img_uj)

    mod = SourceModule("""
                __global__ void HistogramGPU(unsigned char * img,  unsigned char * Hist,const unsigned int height, const unsigned int width)
                {
                    //const unsigned int sor = threadIdx.y + blockIdx.y * blockDim.y;
                    const unsigned int oszlop = threadIdx.x + blockIdx.x * blockDim.x;
                    
                    //__shared__ float shared_kep[360][1280];
                    
                    //if (oszlop ==0)
                    //{
                        //for(int i=0;i<height;i++)
                        //{
                         //   unsigned int idx = oszlop+i*width;
                         //   shared_kep[i][oszlop]= img[idx];
                       // }
                    //}
                    //__syncthreads();
                    
        
                    if (oszlop<width)
                    {
                        int sum=0;
                        for (int i=0; i<height; i++)
                        {
                            unsigned int idx = oszlop+i*width;
                          //  sum+=shared_kep[i][oszlop];
                          sum+=img[idx];
                        }
                         //*(Hist+oszlop)=sum;
                         Hist[oszlop] =sum/width;
                           
                    }
                    else
                    {
                        return;
                    }

                }
        """)

    block_count = int((width - 1) / (BLOCK_SIZE * BLOCK_SIZE) + 1)
    GetHist = mod.get_function("HistogramGPU")
    GetHist(dev_kep, dev_Hist, numpy.uint32(height), numpy.uint32(width), block=(1024, 1, 1),grid=(block_count,1))

    Histogram = numpy.zeros_like(res)
    cuda.memcpy_dtoh(Histogram, dev_Hist)

    return Histogram
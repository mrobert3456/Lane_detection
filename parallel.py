import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
import os

import cv2 as cv
import math

BLOCK_SIZE = 32

if (os.system("cl.exe")):
    os.environ['PATH'] += ';' + r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.28.29910\bin\Hostx64\x64"


def GrayScaleGPU(img):
    result = numpy.empty_like(img)
    #gets the RGB channels
    R = img[:, :, 0].copy()
    G = img[:, :, 1].copy()
    B = img[:, :, 2].copy()

    height = img.shape[0]
    width = img.shape[1]

    #gets the dimensions of the block
    dim_gridx = math.ceil(width / BLOCK_SIZE)
    dim_gridy = math.ceil(height / BLOCK_SIZE)

    #Allocate memory on the gpu
    dev_R = cuda.mem_alloc(R.nbytes)
    dev_G = cuda.mem_alloc(G.nbytes)
    dev_B = cuda.mem_alloc(B.nbytes)

    # copy to gpu
    cuda.memcpy_htod(dev_R, R)
    cuda.memcpy_htod(dev_G, G)
    cuda.memcpy_htod(dev_B, B)

    # Each thread will compute its corresponding cell
    mod = SourceModule("""
            __global__ void Convert_To_Gray(unsigned char * R, unsigned char * G, unsigned char * B, const unsigned int width, const unsigned int height)
            {
                //Calculate indexes of each thread
                const unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
                const unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
                
                __shared__ unsigned char R_shared[1024];
                __shared__  unsigned char G_shared[1024];
                __shared__  unsigned char  B_shared[1024];
                
                //copy data to shared memory
                // each thread copies its corresponding data to the shared memory
                unsigned int idx = col+row*width;
                R_shared[idx%1024] = R[idx%1024];
                G_shared[idx%1024] = G[idx%1024];
                B_shared[idx%1024] = B[idx%1024];
                
                //if the current thread idx is inside the img boundries
                if (row <height && col<width)
                {
                    const unsigned int idx = col+row*width;
                    const unsigned char intensity = R_shared[idx%1024]*0.07+G_shared[idx%1024]*0.72+B_shared[idx%1024]*0.21;

                    R_shared[idx%1024] = intensity;
                    G_shared[idx%1024] = intensity;
                    B_shared[idx%1024] = intensity;
                    
                    // copy data back to global memory
                     R[idx%1024] = R_shared[idx%1024];
                     G[idx%1024] = G_shared[idx%1024];
                     B[idx%1024] = B_shared[idx%1024];
                 }
                else{
                    //if there are threads, which is unnecessary, the it simply returns
                    return;
                }

                 }
       """)

    grayConv = mod.get_function("Convert_To_Gray")

    #determine neccessary block count
    # 1 block can handle max 1024 threads
    #grid =collection of blocks
    # block cannot communicate to each other
    # block have limited shared memory, which is faster than global memory
    block_count = (height * width - 1) / BLOCK_SIZE * BLOCK_SIZE + 1 #921600 threads
    grayConv(dev_R,
             dev_G,
             dev_B,
             numpy.uint32(width),
             numpy.uint32(height),
             block=(BLOCK_SIZE, BLOCK_SIZE, 1), #1 block has 32*32 = 1024 threads
             grid=(dim_gridx, dim_gridy)  # Collection of blocks 40*23 -> so it means that we have 40*23*1024 =942080 threads
             )                            # 942080-921600 = 20480 thread is unnecessary

    # for example:
    # fullhd.jpg shape: 900*1600 so, it has = 1440000 pixels
    # 1440000/1024 = 1407 rounded -> this would be the block count

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
    res = img[0, :] # histogram
    img_uj = img[int(img.shape[0] / 2):, :]
    #shape of the image : 360*1280
    height = img.shape[0] / 2
    width = img.shape[1]

    dev_Hist = cuda.mem_alloc(res.nbytes)
    dev_kep = cuda.mem_alloc(img_uj.nbytes)

    cuda.memcpy_htod(dev_Hist, res)
    cuda.memcpy_htod(dev_kep, img_uj)
    # Each threads sums the corresponding columns
    mod = SourceModule("""
                __global__ void HistogramGPU(unsigned char * img,  unsigned char * Hist,const unsigned int height, const unsigned int width)
                {  
                    const unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
                
                    __shared__ unsigned char shared_img[64*360];
                    __shared__ unsigned char  shared_hist[1280];
                    
                    shared_hist[col%1280] = 0;
                    
                    //copy 64 row to shared memory
                    for (int j=col ;j<(col+64);j++)
                    {
                        for (int i=0; i<360; i++)
                        {
                         unsigned int idx = j+i*width;
                         shared_img[idx%(64*360)] = img[idx];
                         }
                    }
                    __syncthreads();
                     
                    if (col<1280)
                    {
                        unsigned int sum=0;
                        //each thread should sum it's corresponding column
                        for (int i=0; i<360; i++)
                        {
                          unsigned int idx_t = col+i*width;
                          //atomicAdd(&shared_hist[col%1280],shared_img[idx_t%(64*360)]);
                          sum+=int(shared_img[idx_t%(64*360)]);
                        }

                       shared_hist[col%1280] = sum/width;

                        //copy back to global memory
                       __syncthreads();
                       if (col%64==1)
                       {
                            for(int i=(col-63); i<(col+64);i++)
                            {
                                Hist[i%1280]=shared_hist[i%1280];
                            }
                        
                       }
                    }
                    else{
                          return;
                    }

                }
        """)

    block_count = int((width - 1) / (BLOCK_SIZE * BLOCK_SIZE) + 1)
    GetHist = mod.get_function("HistogramGPU")
    GetHist(dev_kep, dev_Hist, numpy.uint32(height), numpy.uint32(width), block=(1024, 1, 1),grid=(block_count,1))

    # copy from gpu to host
    Histogram = numpy.zeros_like(res)
    cuda.memcpy_dtoh(Histogram, dev_Hist)

    return Histogram
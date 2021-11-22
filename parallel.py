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
    block_count = (height * width - 1) / BLOCK_SIZE * BLOCK_SIZE + 1
    grayConv(dev_R,
             dev_G,
             dev_B,
             numpy.uint32(width),
             numpy.uint32(height),
             block=(BLOCK_SIZE, BLOCK_SIZE, 1), #1 block has 32*32 = 1024 threads
             grid=(dim_gridx, dim_gridy)  # Collection of blocks
             )

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
    res = img[0, :]
    img_uj = img[int(img.shape[0] / 2):, :]
    #shape of the image : 360*1280
    height = img.shape[0] / 2
    width = img.shape[1]
    dim_gridx = math.ceil(width / BLOCK_SIZE)
    dim_gridy = math.ceil(height / BLOCK_SIZE)




    dev_Hist = cuda.mem_alloc(res.nbytes)
    dev_kep = cuda.mem_alloc(img_uj.nbytes)

    cuda.memcpy_htod(dev_Hist, res)
    cuda.memcpy_htod(dev_kep, img_uj)
    # Each threads sums the corresponding columns
    mod = SourceModule("""
                __global__ void HistogramGPU(unsigned char * img,  unsigned char * Hist,const unsigned int height, const unsigned int width)
                {
                    //const unsigned int sor = threadIdx.y + blockIdx.y * blockDim.y;
                    const unsigned int oszlop = threadIdx.x + blockIdx.x * blockDim.x;
                    
                    //__shared__ float shared_kep[360][32];
                    
                   // if (oszlop%32 ==0)
                    //{
                      //for(int i=0;i<height;i++)
                        //{
                          //  unsigned int idx = oszlop+i*width;
                           //shared_kep[i][oszlop%32]= img[idx];
                       // }
                    //}
                    //__syncthreads();
                    
        
                    if (oszlop<width)
                    {
                        int sum=0;
                        for (int i=0; i<height; i++)
                        {
                            unsigned int idx = oszlop+i*width;
                            //sum+=shared_kep[i];
                            //atomicAdd(&shared_kep[i][oszlop%32],img[idx]);
                          sum+=img[idx];
                        }
                         //*(Hist+oszlop)=sum;
                        //int seged = Hist[oszlop];
                        //Hist[oszlop] =seged/width;
                       Hist[oszlop] = sum/width;
                           
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
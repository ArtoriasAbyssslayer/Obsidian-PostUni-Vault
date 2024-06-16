```python
!nvcc --version
```

    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2020 NVIDIA Corporation
    Built on Mon_Oct_12_20:09:46_PDT_2020
    Cuda compilation tools, release 11.1, V11.1.105
    Build cuda_11.1.TC455_06.29190527_0
    


```python
!pip -v install git+https://github.com/andreinechaev/nvcc4jupyter.git
```

    Using pip 21.1.3 from /usr/local/lib/python3.7/dist-packages/pip (python 3.7)
    Value for scheme.platlib does not match. Please report this to <https://github.com/pypa/pip/issues/9617>
    distutils: /usr/local/lib/python3.7/dist-packages
    sysconfig: /usr/lib/python3.7/site-packages
    Value for scheme.purelib does not match. Please report this to <https://github.com/pypa/pip/issues/9617>
    distutils: /usr/local/lib/python3.7/dist-packages
    sysconfig: /usr/lib/python3.7/site-packages
    Value for scheme.headers does not match. Please report this to <https://github.com/pypa/pip/issues/9617>
    distutils: /usr/local/include/python3.7/UNKNOWN
    sysconfig: /usr/include/python3.7m/UNKNOWN
    Value for scheme.scripts does not match. Please report this to <https://github.com/pypa/pip/issues/9617>
    distutils: /usr/local/bin
    sysconfig: /usr/bin
    Value for scheme.data does not match. Please report this to <https://github.com/pypa/pip/issues/9617>
    distutils: /usr/local
    sysconfig: /usr
    Additional context:
    user = False
    home = None
    root = None
    prefix = None
    Non-user install because site-packages writeable
    Created temporary directory: /tmp/pip-ephem-wheel-cache-fe0ithos
    Created temporary directory: /tmp/pip-req-tracker-kvy0mkl2
    Initialized build tracking at /tmp/pip-req-tracker-kvy0mkl2
    Created build tracker: /tmp/pip-req-tracker-kvy0mkl2
    Entered build tracker: /tmp/pip-req-tracker-kvy0mkl2
    Created temporary directory: /tmp/pip-install-3kmsh2bc
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting git+https://github.com/andreinechaev/nvcc4jupyter.git
      Created temporary directory: /tmp/pip-req-build-8dr63rqn
      Cloning https://github.com/andreinechaev/nvcc4jupyter.git to /tmp/pip-req-build-8dr63rqn
      Running command git clone -q https://github.com/andreinechaev/nvcc4jupyter.git /tmp/pip-req-build-8dr63rqn
      Added git+https://github.com/andreinechaev/nvcc4jupyter.git to build tracker '/tmp/pip-req-tracker-kvy0mkl2'
        Running setup.py (path:/tmp/pip-req-build-8dr63rqn/setup.py) egg_info for package from git+https://github.com/andreinechaev/nvcc4jupyter.git
        Created temporary directory: /tmp/pip-pip-egg-info-0_otbeig
        Running command python setup.py egg_info
        running egg_info
        creating /tmp/pip-pip-egg-info-0_otbeig/NVCCPlugin.egg-info
        writing /tmp/pip-pip-egg-info-0_otbeig/NVCCPlugin.egg-info/PKG-INFO
        writing dependency_links to /tmp/pip-pip-egg-info-0_otbeig/NVCCPlugin.egg-info/dependency_links.txt
        writing top-level names to /tmp/pip-pip-egg-info-0_otbeig/NVCCPlugin.egg-info/top_level.txt
        writing manifest file '/tmp/pip-pip-egg-info-0_otbeig/NVCCPlugin.egg-info/SOURCES.txt'
        writing manifest file '/tmp/pip-pip-egg-info-0_otbeig/NVCCPlugin.egg-info/SOURCES.txt'
      Source in /tmp/pip-req-build-8dr63rqn has version 0.0.2, which satisfies requirement NVCCPlugin==0.0.2 from git+https://github.com/andreinechaev/nvcc4jupyter.git
      Removed NVCCPlugin==0.0.2 from git+https://github.com/andreinechaev/nvcc4jupyter.git from build tracker '/tmp/pip-req-tracker-kvy0mkl2'
    Created temporary directory: /tmp/pip-unpack-qpkzpkco
    Building wheels for collected packages: NVCCPlugin
      Created temporary directory: /tmp/pip-wheel-kgowx_74
      Building wheel for NVCCPlugin (setup.py) ... [?25l  Destination directory: /tmp/pip-wheel-kgowx_74
      Running command /usr/bin/python3 -u -c 'import io, os, sys, setuptools, tokenize; sys.argv[0] = '"'"'/tmp/pip-req-build-8dr63rqn/setup.py'"'"'; __file__='"'"'/tmp/pip-req-build-8dr63rqn/setup.py'"'"';f = getattr(tokenize, '"'"'open'"'"', open)(__file__) if os.path.exists(__file__) else io.StringIO('"'"'from setuptools import setup; setup()'"'"');code = f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' bdist_wheel -d /tmp/pip-wheel-kgowx_74
      running bdist_wheel
      running build
      running build_py
      creating build
      creating build/lib
      copying nvcc_plugin.py -> build/lib
      creating build/lib/v2
      copying v2/__init__.py -> build/lib/v2
      copying v2/v2.py -> build/lib/v2
      creating build/lib/v1
      copying v1/__init__.py -> build/lib/v1
      copying v1/v1.py -> build/lib/v1
      creating build/lib/common
      copying common/__init__.py -> build/lib/common
      copying common/helper.py -> build/lib/common
      installing to build/bdist.linux-x86_64/wheel
      running install
      running install_lib
      creating build/bdist.linux-x86_64
      creating build/bdist.linux-x86_64/wheel
      creating build/bdist.linux-x86_64/wheel/v2
      copying build/lib/v2/__init__.py -> build/bdist.linux-x86_64/wheel/v2
      copying build/lib/v2/v2.py -> build/bdist.linux-x86_64/wheel/v2
      creating build/bdist.linux-x86_64/wheel/v1
      copying build/lib/v1/v1.py -> build/bdist.linux-x86_64/wheel/v1
      copying build/lib/v1/__init__.py -> build/bdist.linux-x86_64/wheel/v1
      creating build/bdist.linux-x86_64/wheel/common
      copying build/lib/common/helper.py -> build/bdist.linux-x86_64/wheel/common
      copying build/lib/common/__init__.py -> build/bdist.linux-x86_64/wheel/common
      copying build/lib/nvcc_plugin.py -> build/bdist.linux-x86_64/wheel
      running install_egg_info
      running egg_info
      creating NVCCPlugin.egg-info
      writing NVCCPlugin.egg-info/PKG-INFO
      writing dependency_links to NVCCPlugin.egg-info/dependency_links.txt
      writing top-level names to NVCCPlugin.egg-info/top_level.txt
      writing manifest file 'NVCCPlugin.egg-info/SOURCES.txt'
      writing manifest file 'NVCCPlugin.egg-info/SOURCES.txt'
      Copying NVCCPlugin.egg-info to build/bdist.linux-x86_64/wheel/NVCCPlugin-0.0.2-py3.7.egg-info
      running install_scripts
      creating build/bdist.linux-x86_64/wheel/NVCCPlugin-0.0.2.dist-info/WHEEL
      creating '/tmp/pip-wheel-kgowx_74/NVCCPlugin-0.0.2-py3-none-any.whl' and adding 'build/bdist.linux-x86_64/wheel' to it
      adding 'nvcc_plugin.py'
      adding 'common/__init__.py'
      adding 'common/helper.py'
      adding 'v1/__init__.py'
      adding 'v1/v1.py'
      adding 'v2/__init__.py'
      adding 'v2/v2.py'
      adding 'NVCCPlugin-0.0.2.dist-info/METADATA'
      adding 'NVCCPlugin-0.0.2.dist-info/WHEEL'
      adding 'NVCCPlugin-0.0.2.dist-info/top_level.txt'
      adding 'NVCCPlugin-0.0.2.dist-info/RECORD'
      removing build/bdist.linux-x86_64/wheel
    [?25hdone
      Created wheel for NVCCPlugin: filename=NVCCPlugin-0.0.2-py3-none-any.whl size=4306 sha256=119985d8725abcf2e47e24d707d16c3a41c5087fcea9fc458a7b6b17d779ddfb
      Stored in directory: /tmp/pip-ephem-wheel-cache-fe0ithos/wheels/ca/33/8d/3c86eb85e97d2b6169d95c6e8f2c297fdec60db6e84cb56f5e
    Successfully built NVCCPlugin
    Installing collected packages: NVCCPlugin
      Value for scheme.platlib does not match. Please report this to <https://github.com/pypa/pip/issues/9617>
      distutils: /usr/local/lib/python3.7/dist-packages
      sysconfig: /usr/lib/python3.7/site-packages
      Value for scheme.purelib does not match. Please report this to <https://github.com/pypa/pip/issues/9617>
      distutils: /usr/local/lib/python3.7/dist-packages
      sysconfig: /usr/lib/python3.7/site-packages
      Value for scheme.headers does not match. Please report this to <https://github.com/pypa/pip/issues/9617>
      distutils: /usr/local/include/python3.7/NVCCPlugin
      sysconfig: /usr/include/python3.7m/NVCCPlugin
      Value for scheme.scripts does not match. Please report this to <https://github.com/pypa/pip/issues/9617>
      distutils: /usr/local/bin
      sysconfig: /usr/bin
      Value for scheme.data does not match. Please report this to <https://github.com/pypa/pip/issues/9617>
      distutils: /usr/local
      sysconfig: /usr
      Additional context:
      user = False
      home = None
      root = None
      prefix = None
      Running command git rev-parse HEAD
      aac710a35f52bb78ab34d2e52517237941399eff
    
    Value for scheme.platlib does not match. Please report this to <https://github.com/pypa/pip/issues/9617>
    distutils: /usr/local/lib/python3.7/dist-packages
    sysconfig: /usr/lib/python3.7/site-packages
    Value for scheme.purelib does not match. Please report this to <https://github.com/pypa/pip/issues/9617>
    distutils: /usr/local/lib/python3.7/dist-packages
    sysconfig: /usr/lib/python3.7/site-packages
    Value for scheme.headers does not match. Please report this to <https://github.com/pypa/pip/issues/9617>
    distutils: /usr/local/include/python3.7/UNKNOWN
    sysconfig: /usr/include/python3.7m/UNKNOWN
    Value for scheme.scripts does not match. Please report this to <https://github.com/pypa/pip/issues/9617>
    distutils: /usr/local/bin
    sysconfig: /usr/bin
    Value for scheme.data does not match. Please report this to <https://github.com/pypa/pip/issues/9617>
    distutils: /usr/local
    sysconfig: /usr
    Additional context:
    user = False
    home = None
    root = None
    prefix = None
    Successfully installed NVCCPlugin-0.0.2
    Removed build tracker: '/tmp/pip-req-tracker-kvy0mkl2'
    


```python
%load_ext nvcc_plugin
```

    created output directory at /content/src
    Out bin /content/result.out
    


```python
%%cu 

#include <iostream> 

int main() { 
    //Here simple c program is only executed.
    //All complex CUDA program can be executed in the Google colab environment using this way.
    printf("CUDA is working\n");
    return 0; 
}
```

    CUDA is working
    
    


```python
%%cu
#include <stdio.h>
#include <stdlib.h>
#include <random>
const int IH=512;
const int IW=512;
const int MH=3;
const int MW=3;
const int arraySize = IH * IW;
const int maskSize = MH * MW;
const int TILE=32;

__global__ void conv(float* OUT, float* IN, float* M, int inw, int inh, int mw, int mh)
{
    /*Get row and column to operate on from thread coordinates*/
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    /*Calculate "padding" radius of convolution kernel (distance around central pixel)*/
    int pw = (mw - 1) / 2;
    int ph = (mh - 1) / 2;
    __syncthreads();
 
    /*NOT THIS If within the range of OUT (ie IN - padding) NOT THIS */
    // if (row < (inh - 2 * ph) && col < (inw - 2 * pw))  //NO PADDING - SHRINKED OUT IMAGE
    // if (row < (inh) && col < (inw)) 

    /*Set initial pixel value*/
    float val = 0.0f;
    for (int i = -ph; i <= ph; i = i + 1)
    {
        for (int j = -pw; j <= pw; j = j + 1) 
        {
            // Add product of kernel value and corresponding image value to running total
            // ph-i and pw-j flips both horizontally and vertically the image tile
            // ph+i and pw+j shifts mask coordinates to a [0,0] basis
            val += IN[(row + ph - i) * inw + (col + pw - j)] * M[(i+ph) * mw + (j+pw)];
        }
     
    
    }
    /*Copy resulting pixel value to position on OUT matrix*/
      OUT[row * inw + col] = val;
      __syncthreads();
}

// 2D CONVOLUTION function BETWEEN AN IMAGE AND A MASK. DO THIS FIRST
cudaError_t convolution(float* out, float* in, float* m)
{
    float* dev_out = 0;
    float* dev_in = 0;
    float* dev_m;
    cudaError_t cudaStatus; //debugging object

    // Launch a kernel on the GPU with 
    //Grid number of blocks and Block number of threads for each element.
    
    dim3 Grid(IW / TILE, IH / TILE, 1);
    // dim3 Grid(16, 16, 1);
    dim3 Block(TILE, TILE, 1);
    // dim3 Block(32, 32, 1);
  
    // Allocate GPU memory for three vectors (two input, one output)    
    cudaMalloc((void**)& dev_out, IH * IW * sizeof(float));
    
    cudaMalloc((void**)& dev_in, IH * IW * sizeof(float));
    
    cudaMalloc((void**)& dev_m, MH * MW * sizeof(float));
    
    // Copy input vectors (image and maks) from host memory to GPU buffers.
    cudaMemcpy(dev_in, in, IH * IW * sizeof(/*?*/ float), cudaMemcpyHostToDevice);
    
    cudaMemcpy(dev_m, m, MH * MW * sizeof(float), cudaMemcpyHostToDevice);
    
    conv <<<Grid, Block>>> (dev_out, dev_in, dev_m, IW, IH,MW,MH);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
      fprintf(stderr, "conv2D kernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); //fprintf for formatted output
      goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
      fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching CONVOLUTION Kernel!\n", cudaStatus);
      goto Error;
    }

    cudaMemcpy(out, dev_out, IH * IW * sizeof(float), cudaMemcpyDeviceToHost);
  
    Error:	
      cudaFree(dev_out);
      cudaFree(dev_in);
      cudaFree(dev_m);

      return cudaStatus;
}

int main()
{
  // dynamically allocate space in the cpu for an input_image array, that essentially will be a 512 times 512 cells 1D vector with random values to avoid reading an image with the colab c extension
	float* input_image = (float *)malloc(sizeof(float) * arraySize); 
  //float* mask = (float *)malloc(sizeof(float) * maskSize);
 
 // dynamically allocating space in the cpu to copy the results of the kernel
  float* convolved_image = (float *)malloc(sizeof(float) * arraySize);

  float mask[maskSize] = { 1/9.0f, 1 / 9.0f, 1 / 9.0f, 1 / 9.0f, 1 / 9.0f, 1 / 9.0f, 1 / 9.0f, 1 / 9.0f, 1 / 9.0f };
	//float mask[maskSize] = { 1.0f, 1.0f, 1.0f, 1.0f, -8.0f, 1.0f, 1.0f, 1.0f, 1.0f };
  //float mask[maskSize] = { 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, -1.0f, -1.0f };
  //float mask[maskSize] = { 1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f };
  for(int i=0;i<arraySize;i++)
  {
    //This is a random integer in the range of 0 to 255 in C++
		input_image[i]=(rand() % 256); 
  }
	printf("INPUT IMAGE (10x10 tile top left corner)\n");
	for(int i=0;i<10;i++)
  {
    for(int j=0;j<10;j++)
    {
       printf("%.3f ", input_image[i*IW+j]);
    }
       printf("\n");
 	}
  convolution(convolved_image, input_image, mask);
	printf("FILTERED IMAGE (10x10 tile top left corner)\n");
 	for(int i=0;i<10;i++)
  {
    for(int j=0;j<10;j++)
    {
       printf("%.3f ", convolved_image[i*IW+j]);
    }
    printf("\n");
 	}
 
 
 return 0;
}

```

    INPUT IMAGE (10x10 tile top left corner)
    103.000 198.000 105.000 115.000 81.000 255.000 74.000 236.000 41.000 205.000 
    149.000 170.000 130.000 202.000 108.000 73.000 174.000 144.000 205.000 22.000 
    43.000 140.000 182.000 135.000 27.000 100.000 245.000 97.000 171.000 28.000 
    161.000 1.000 228.000 217.000 168.000 89.000 37.000 49.000 199.000 154.000 
    183.000 244.000 136.000 121.000 44.000 240.000 189.000 132.000 254.000 145.000 
    34.000 179.000 48.000 62.000 194.000 88.000 251.000 18.000 124.000 183.000 
    193.000 87.000 224.000 48.000 140.000 5.000 221.000 239.000 157.000 125.000 
    165.000 8.000 223.000 86.000 149.000 52.000 62.000 150.000 210.000 102.000 
    177.000 212.000 88.000 89.000 218.000 216.000 179.000 90.000 24.000 94.000 
    228.000 49.000 251.000 96.000 193.000 172.000 189.000 213.000 231.000 190.000 
    FILTERED IMAGE (10x10 tile top left corner)
    135.556 153.000 120.556 121.778 126.333 155.333 154.111 127.667 132.556 141.889 
    133.778 156.111 155.222 124.333 113.444 112.000 146.778 118.778 132.222 118.556 
    146.444 156.000 139.778 126.778 126.556 130.889 152.556 136.556 147.111 115.889 
    134.889 137.333 135.333 135.889 144.444 121.444 139.222 139.778 150.333 130.778 
    147.556 127.667 113.000 104.667 152.444 153.667 176.111 153.000 136.556 126.444 
    129.000 107.222 130.444 91.556 129.111 120.667 159.111 145.333 123.333 133.556 
    153.000 118.333 140.556 111.444 138.000 134.889 148.000 132.333 103.111 109.111 
    155.667 122.444 154.778 141.222 158.889 147.000 149.778 144.889 120.000 125.778 
    139.111 122.000 147.222 175.444 183.000 173.667 136.222 123.222 114.444 119.222 
    114.778 88.778 131.222 160.889 174.111 171.556 139.111 136.111 130.222 150.222 
    
    


```python

```

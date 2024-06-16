__global__ void conv(const float *IN, const float* __restrict__ M, int inw, int inh, int mw, int mh, float *OUT){

    /*Get row and column to operate on from thread coordinates*/
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int row = by*blockDim.y + ty;
    int col = bx*blockDim.x + tx;
    
    /*Calculate "padding" radius of convolution kernel (distance around central pixel)*/
    int pw = (mw-1)/2;
    int ph = (mh-1)/2;

    /*If within the range of OUT (ie IN - padding)*/
    if( row < (inh-2*ph) && col < (inw-2*pw) ) {
        
        /*Set initial pixel value*/
        int val = 0;
        
        /*For each vertical position on the kernel matrix, relative to the central pixel*/
        for(int i=-ph; i<=ph; i=i+1){
            /*Calculate zero-indexed row ID on kernel matrix*/
            int b_row = i+ph; 

            /*For each horizontal position on the kernel matrix, relative to the central pixel*/
            for(int j=-pw; j<=pw; j=j+1){
                /*Calculate zero-indexed column ID on kernel matrix*/
                int b_col = j+pw;

                /*Add product of kernel value and corresponding image value to running total*/
                val += IN[ (row+ph -i)*inw + (col+pw -j) ] * M[ b_row*mw + b_col ];
            }
        }
        
        /*Copy resulting pixel value to position on OUT matrix*/
        OUT[row*(inw-2*pw) + col] = val;
    }
}
from re import M
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.driver import Event
import math

module = SourceModule("""
#include <stdio.h>

__global__ void MovingAverage(float *prices, int lenght, int days, float* out){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float sh[];
    
    sh[threadIdx.x] = prices[tid];
    __syncthreads();
    
    float sum = 0.0f;
    
    if(lenght > lenght - days && tid <= lenght - days) {
        for (int day_shift = tid; day_shift < tid + days; ++day_shift){
            sum = sum + prices[day_shift];
        }
        out[tid] = sum / days;
        tid += 1;
    }
}

__global__ void TrendLine(float* upTrend, float* downTrend, int lenght, int start, int* trendLine) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ((tid >= start) && (tid < lenght)) {
        if (upTrend[tid] >= downTrend[tid - start]){
            trendLine[tid] = 1;
        } else if (upTrend[tid] < downTrend[tid - start]){
        trendLine[tid] = 2-3;
        }
    } else if (tid < start) {
        trendLine[tid] = 0;
    }
    tid += 1;
}
""")


def get_trend(prices: np.ndarray) -> np.ndarray:
    # MovingAverage
    MovingAverage = module.get_function('MovingAverage')
    h = np.array(prices, dtype=np.float32)
    d = cuda.mem_alloc(h.nbytes)
    cuda.memcpy_htod(d, h)
    d_diff = cuda.mem_alloc(h.nbytes)
    block_size_ = 256
    grid_ = math.ceil(len(h) / block_size_)

    # ma 15
    ma_6 = 15
    MovingAverage(d, np.int32(len(h)), np.int32(ma_6), d_diff, block=(block_size_, 1, 1), grid=(grid_, 1, 1),
                  shared=block_size_ * grid_)
    h_ma6 = np.copy(h)
    cuda.memcpy_dtoh(h_ma6, d_diff)
    h_ma6 = h_ma6[:len(h_ma6) - ma_6 + 1]

    # ma 30
    ma_12 = 30
    MovingAverage(d, np.int32(len(h)), np.int32(ma_12), d_diff, block=(block_size_, 1, 1), grid=(grid_, 1, 1),
                  shared=block_size_ * grid_)
    h_ma12 = np.copy(h)
    cuda.memcpy_dtoh(h_ma12, d_diff)
    h_ma12 = h_ma12[:len(h_ma12) - ma_12 + 1]

    # TrendLine
    TrendLine = module.get_function('TrendLine')

    # uptrend
    h_uptrend = np.array(h_ma6, dtype=np.float32)
    d_uptrend = cuda.mem_alloc(h_uptrend.nbytes)
    cuda.memcpy_htod(d_uptrend, h_uptrend)

    # downtrend
    h_downtrend = np.array(h_ma12, dtype=np.float32)
    d_downtrend = cuda.mem_alloc(h_downtrend.nbytes)
    cuda.memcpy_htod(d_downtrend, h_downtrend)

    # trendline
    h_trendLine = h_trendLine = np.ones(len(h_ma6), dtype=np.float32)
    d_trendLine = cuda.mem_alloc(h_trendLine.nbytes)
    cuda.memcpy_htod(d_trendLine, h_trendLine)

    # other definitions
    start = len(h_ma6) - len(h_ma12)
    lenght = len(h_ma6)
    grid_ = math.ceil(len(h_ma6) / block_size_)

    TrendLine(d_uptrend, d_downtrend, np.int32(lenght), np.int32(start), d_trendLine, block=(block_size_, 1, 1),
              grid=(grid_, 1, 1))

    h_result = np.ones(len(h_uptrend), dtype=np.float32)
    cuda.memcpy_dtoh(h_result, d_trendLine)

    # Костыльно, но когда TrendLine отдает значения тренда то вместо 1 пишет 1.4..e-45, а вместо -1 пишет nan
    h_result = np.ndarray.tolist(h_result)
    h_result = [str(x) for x in h_result]
    for i in range(len(h_result)):
        if h_result[i] == '0.0':
            h_result[i] = 0
        elif h_result[i] == '1.401298464324817e-45':
            h_result[i] = 1
        else:
            h_result[i] = -1

    return np.array(h_result)


if __name__ == '__main__':
    prices = np.random.rand(300)
    print(get_trend(prices))
    pass

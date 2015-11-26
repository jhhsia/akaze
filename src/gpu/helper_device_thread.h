//
// Created by prenav on 11/23/15.
//

#ifndef AKAZE_HEALPER_DEVICE_THREAD_H
#define AKAZE_HEALPER_DEVICE_THREAD_H

#pragma once

__device__
inline int GetGlobalIdx(){
    int blockId  = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y)
                   + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}


#endif //AKAZE_HEALPER_DEVICE_THREAD_H

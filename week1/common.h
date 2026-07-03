#ifndef COMMON_H
#define COMMON_H

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// CUDA 错误检查
//
// 类比:嵌入式里你会给每个 HAL 调用包一层返回码检查(if (ret != HAL_OK) ...)。
// CUDA runtime API 也返回 cudaError_t,但绝大多数人懒得每次都写检查,出了错
// 就得到一个莫名其妙的结果。这个宏就是强制你"每次调用都检查",出错时立刻
// 打印文件 + 行号 + 错误名并退出。
//
// 用法:CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d in '%s': %s (%d)\n",          \
                    __FILE__, __LINE__, #call,                                 \
                    cudaGetErrorString(err__), (int)err__);                    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel 启动错误检查
//
// 关键区别:kernel 启动是异步的,启动本身的配置错误(比如 block 太大)通过
// cudaGetLastError() 拿到;而 kernel 执行期间的错误(比如越界访问触发的
// 非法内存访问)要等到某个同步点才暴露,所以这里跟一个 cudaDeviceSynchronize()。
//
// 这个宏在开发/调试时每次 kernel 后都调用。注意:cudaDeviceSynchronize() 会
// 强制 host 等 device 跑完,会掩盖掉异步带来的性能收益 —— 所以正式测性能时
// 应该把它去掉或只在 debug 编译里开。
//
// 用法:
//   myKernel<<<grid, block>>>(...);
//   CUDA_CHECK_KERNEL();
// ---------------------------------------------------------------------------
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        cudaError_t err__ = cudaGetLastError();                               \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "Kernel launch error at %s:%d: %s (%d)\n",         \
                    __FILE__, __LINE__,                                        \
                    cudaGetErrorString(err__), (int)err__);                    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
        err__ = cudaDeviceSynchronize();                                      \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "Kernel exec error at %s:%d: %s (%d)\n",           \
                    __FILE__, __LINE__,                                        \
                    cudaGetErrorString(err__), (int)err__);                    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// 简单计时:用 CUDA event 给一段 GPU 工作计时(单位 ms)
//
// 类比:类似你在裸机上用硬件定时器给一段代码打时间戳,只是这里的"定时器"
// 是插进 GPU 命令流里的 event,测的是 GPU 实际执行的墙钟时间。
//
// 用法:
//   cudaEvent_t start, stop;
//   timer_start(&start, &stop);
//   myKernel<<<...>>>(...);
//   float ms = timer_stop(start, stop);
// ---------------------------------------------------------------------------
static inline void timer_start(cudaEvent_t *start, cudaEvent_t *stop) {
    CUDA_CHECK(cudaEventCreate(start));
    CUDA_CHECK(cudaEventCreate(stop));
    CUDA_CHECK(cudaEventRecord(*start, 0));
}

static inline float timer_stop(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

#endif // COMMON_H

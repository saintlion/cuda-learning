// week1: vector add (PMPP ch.2)
// 编译: nvcc -arch=sm_75 -O2 -lineinfo vecadd.cu -o vecadd
#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ---------------------------------------------------------------------------
// TODO(你来写): vector add kernel
//
// 要求:
//   - 每个线程负责输出向量的一个元素
//   - 必须有边界检查(N 不一定是 blockDim 的整数倍,最后一个 block
//     里会有多余的线程,它们不能越界写)
//
// 提示:全局线程索引由 blockIdx / blockDim / threadIdx 组合出来。
// ---------------------------------------------------------------------------
__global__ void vecAddKernel(const float *a, const float *b, float *c, int n) {
    int index = blockDim.x*blockIdx.x+threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// CPU 参考实现,用来验证 GPU 结果
static void vecAddCPU(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// 逐元素比对 GPU 和 CPU 结果。float 加法这里没有累积误差,理论上应该
// bit 级一致,但习惯上仍用一个小 epsilon 比较。
static bool verify(const float *gpu, const float *ref, int n) {
    for (int i = 0; i < n; ++i) {
        if (fabsf(gpu[i] - ref[i]) > 1e-5f) {
            fprintf(stderr, "MISMATCH at %d: gpu=%f ref=%f\n", i, gpu[i], ref[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // 故意选一个不是 256 整数倍的 N,这样边界检查缺失会立刻暴露
    const int N = (1 << 20) + 3;
    const size_t bytes = N * sizeof(float);

    // ---- host 端内存分配与初始化 ----
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);   // GPU 结果拷回来放这里
    float *h_ref = (float *)malloc(bytes); // CPU 参考结果
    if (!h_a || !h_b || !h_c || !h_ref) {
        fprintf(stderr, "host malloc failed\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)(i % 1000) * 0.5f;
        h_b[i] = (float)((N - i) % 1000) * 0.25f;
    }

    // ---- device 端内存分配 ----
    // 类比:d_* 是"外设自己的 RAM",host 指针不能直接解引用它
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // ---- host -> device 拷贝(类比:CPU 内存 DMA 到外设内存)----
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // ---- 启动 kernel ----
    // grid 大小用向上取整除法:保证线程总数 >= N,多出来的靠边界检查挡掉
    const int block = 256;
    const int grid = (N + block - 1) / block;

    cudaEvent_t start, stop;
    timer_start(&start, &stop);
    vecAddKernel<<<grid, block>>>(d_a, d_b, d_c, N);
    float ms = timer_stop(start, stop);
    CUDA_CHECK_KERNEL(); // 检查启动配置错误 + 同步后检查执行期错误

    // ---- device -> host 拷回结果 ----
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // ---- CPU 验证 ----
    vecAddCPU(h_a, h_b, h_ref, N);
    bool ok = verify(h_c, h_ref, N);

    printf("N = %d, grid = %d x %d threads\n", N, grid, block);
    printf("kernel time: %.3f ms\n", ms);
    printf("verify: %s\n", ok ? "PASS" : "FAIL");

    // ---- 清理 ----
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_ref);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

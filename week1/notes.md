# Week 1 笔记

## vector add (PMPP ch.2)

文件:`vecadd.cu`

### 踩的坑(按顺序)

1. **第一版只用了 `threadIdx.x`,完全没碰 `blockIdx`**:

   ```cpp
   int index = threadIdx.x;
   int stride = blockDim.x;
   for (int i = index; i < n; i += stride) { ... }
   ```

   问题:不同 block 里 `threadIdx.x` 相同的线程会算出完全一样的
   `index`/`stride`,导致所有 block 重复做同一份工作(只有 block 0
   真正有效,其余全是无意义的重复劳动 + 对同一地址的重复写)。

2. **疑问:"一次循环加上 stride,不是有好多次加法运算被跳过了吗?"**

   卡住的点:以为单个线程循环里 `i += stride` 跳过的那些下标就没人处理了。
   后来想通:跳过的部分是**交给了别的线程**——不同 `threadIdx` 的线程
   起始 `index` 不同,大家交错(round-robin)分工,合起来正好覆盖全部
   下标,没有遗漏也没有重复。

3. **改的时候两个语法错误**:
   - `blockDim.x * blockIdx` —— 少打了 `.x`,`blockIdx` 是 `dim3`
     结构体不是整数,不能直接乘。
   - `stride = blockDim.x + gridDim.x` —— 该用乘法(`blockDim.x
     * gridDim.x`,算出全局线程总数),写成了加法。

4. **最终版本**:

   ```cpp
   int index = blockDim.x * blockIdx.x + threadIdx.x;   // 全局线程编号
   int stride = blockDim.x * gridDim.x;                  // 全局线程总数
   for (int i = index; i < n; i += stride) {
       c[i] = a[i] + b[i];
   }
   ```

### 概念澄清

- CUDA 里并发执行的单位叫**线程(thread)**,不是"进程"——线程间共享
  同一份全局内存(`a`/`b`/`c` 指向的显存),只是各自有私有的局部变量
  和身份标识(`threadIdx`/`blockIdx`)。这是 SPMD(单程序多数据)模型:
  同一份 kernel 代码被实例化成 `grid * block` 份并发执行,靠
  `threadIdx`/`blockIdx` 区分身份。
- `blockDim`:block 尺寸(每个 block 有多少线程,所有 block 一致)。
- `threadIdx`:线程在**所在 block 内**的局部编号。
- `blockIdx`:block 在整个 grid 里的编号。
- 硬件调度是以 block 为粒度的,同一 block 的线程会被分到同一个 SM,
  可以共享 shared memory、用 `__syncthreads()` 同步;block 间调度顺序
  不保证。
- warp(32 线程一组)是 lockstep 执行的,分支不一致会有 warp divergence
  开销——这点和 RTOS 任务调度的"类比"在这里失效。

### 运行结果

环境:Colab 免费额度用尽,改用 Kaggle(2x T4)。

```
N = 1048579, grid = 4097 x 256 threads
kernel time: 0.564 ms
verify: PASS
```

- `N = 1048579 = (1<<20) + 3`,故意不是 256 的整数倍,用来验证边界检查
  (grid-stride loop 的循环条件 `i < n` 本身就是边界检查)。
- PASS 说明最后一个 block 里多出来的线程没有越界写。
- 0.564ms 量级合理:向量加法是纯内存带宽瓶颈,不是算力瓶颈。

### 环境笔记

- Colab 免费额度不定期恢复(约 12~24 小时),Pay-As-You-Go 约 $9.99/100
  计算单元,T4 约 1.8~2 CU/小时(约 $0.18~0.20/小时),不过期,适合这种
  偶尔跑几分钟的小实验。
- 备用方案:Kaggle 提供免费 2x T4。

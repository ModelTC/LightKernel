/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#include "cutlass/fast_math.h"
#include "cutlass/barrier.h"
#include "cutlass/arch/barrier.h"

#include "cutlass/arch/grid_dependency_control.h"

#include "flash.h"

namespace flash {

/**
 * 为可变长度序列准备动态分割数的GPU核函数
 * 这个函数为每个批次计算最优的分割数，用于负载均衡
 * 
 * @param seqlen_q_static Q序列的静态长度（当没有变长时使用）
 * @param seqlen_k_static K序列的静态长度（当没有变长时使用）
 * @param seqlen_k_new_static 新K序列的静态长度
 * @param cu_seqlens_q Q序列的累积长度数组（变长序列）
 * @param cu_seqlens_k K序列的累积长度数组（变长序列）
 * @param cu_seqlens_k_new 新K序列的累积长度数组
 * @param seqused_q 实际使用的Q序列长度数组
 * @param seqused_k 实际使用的K序列长度数组
 * @param leftpad_k_ptr K序列的左侧填充长度数组
 * @param num_batch 批次数量
 * @param num_head 头数
 * @param qhead_per_khead 每个K头对应的Q头数（用于GQA）
 * @param num_sm SM数量
 * @param num_splits_static 静态分割数（启发式计算得出）
 * @param blockm_divmod M维度的块分割器
 * @param blockn_divmod N维度的块分割器
 * @param tile_count_semaphore 瓦片计数信号量
 * @param num_splits_dynamic_ptr 输出：每个批次的动态分割数数组
 * @param enable_pdl 是否启用程序依赖启动（PDL）
 */
__global__ void prepare_varlen_num_blocks_kernel(
        int seqlen_q_static, int seqlen_k_static, int seqlen_k_new_static,
        int const* const cu_seqlens_q, int const* const cu_seqlens_k, int const* const cu_seqlens_k_new,
        int const* const seqused_q, int const* const seqused_k, int const* const leftpad_k_ptr,
        int num_batch, int num_head, int qhead_per_khead, int num_sm, int num_splits_static,
        cutlass::FastDivmod blockm_divmod, cutlass::FastDivmod blockn_divmod,
        int* const tile_count_semaphore,
        // int* const num_m_blocks_ptr,
        int* const num_splits_dynamic_ptr,
        bool enable_pdl) {

    // 每个warp处理的批次数量（32-1=31个，预留一个线程用于shuffle操作）
    static constexpr int kNumBatchPerWarp = cutlass::NumThreadsPerWarp - 1;
    // 共享内存大小（用于存储总块数）
    static constexpr int kSmemSize = 1;
    // 网格中只有一个线程块，假设这一点
    __shared__ int total_blocks_smem[kSmemSize];

    // 网格中只有1个块，因此可以启动主要的注意力核函数
    // PDL（Program Dependent Launch）允许在当前核函数运行时启动依赖的核函数
    if (enable_pdl) { cutlass::arch::launch_dependent_grids(); }

    // 初始化共享内存为0
    if (threadIdx.x < kSmemSize) { total_blocks_smem[threadIdx.x] = 0; }
    __syncthreads();

    // 线程0初始化瓦片计数信号量
    if (threadIdx.x == 0 && tile_count_semaphore) { *tile_count_semaphore = 0; }

    // 获取当前线程在warp中的lane ID
    int lane = threadIdx.x % cutlass::NumThreadsPerWarp;

    // Lambda函数：计算M维度（Q序列方向）的块数量
    auto get_num_m_blocks = [&](int bidb_start) {
        // 当前处理的批次索引
        int batch_idx = lane + bidb_start;
        int seqlen;
        
        // 根据不同的输入格式获取Q序列长度
        if (seqused_q) {
            // 直接使用实际序列长度数组
            seqlen = batch_idx < num_batch ? seqused_q[batch_idx] : 0;
        } else if (cu_seqlens_q) {
            // 使用累积长度数组计算当前批次的序列长度
            int cur_cu_seqlen = batch_idx <= num_batch ? cu_seqlens_q[batch_idx] : 0;
            int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
            seqlen = next_cu_seqlen - cur_cu_seqlen;
        } else {
            // 使用静态序列长度
            seqlen = seqlen_q_static;
        }
        
        // 对于GQA（Grouped Query Attention），需要乘以Q头与K头的比例 16
        seqlen *= qhead_per_khead;
        
        // 计算需要的M维度块数（向上取整）
        // ceil(seqlen / blockM) = ceil(16 / 64) = 1
        return batch_idx < num_batch && lane < kNumBatchPerWarp
            ? blockm_divmod.div(seqlen + blockm_divmod.divisor - 1) : 0;
    };

    // Lambda函数：计算N维度（K序列方向）的块数量
    auto get_num_n_blocks = [&](int bidb_start) {
        int batch_idx = lane + bidb_start;  // 当前处理的批次索引
        
        // 获取左侧填充长度 0
        int leftpad_k = batch_idx < num_batch && leftpad_k_ptr != nullptr ? leftpad_k_ptr[batch_idx] : 0;
        
        int seqlen;
        // 根据不同的输入格式获取K序列长度
        if (seqused_k) {
            // 直接使用实际序列长度数组
            seqlen = batch_idx < num_batch ? seqused_k[batch_idx] : 0;
        } else if (cu_seqlens_k) {
            // 使用累积长度数组计算当前批次的序列长度
            int cur_cu_seqlen = batch_idx <= num_batch ? cu_seqlens_k[batch_idx] : 0;
            int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
            seqlen = next_cu_seqlen - cur_cu_seqlen;
        } else {
            // 使用静态序列长度
            seqlen = seqlen_k_static;
        }
        
        // 获取新K序列的长度（用于KV缓存扩展场景）
        // cu_seqlens_k_new 为空因为没传 k_new
        int seqlen_new;
        if (cu_seqlens_k_new) {
            int cur_cu_seqlen_new = batch_idx <= num_batch ? cu_seqlens_k_new[batch_idx] : 0;
            int next_cu_seqlen_new = __shfl_down_sync(0xffffffff, cur_cu_seqlen_new, 1);
            seqlen_new = next_cu_seqlen_new - cur_cu_seqlen_new;
            // if (threadIdx.x == 0) { printf("cur_cu_seqlen_new=%d, next_cu_seqlen_new=%d, seqlen = %d, seqlen_new = %d, leftpad_k = %d\n", cur_cu_seqlen_new, next_cu_seqlen_new, seqlen, seqlen_new, leftpad_k); }
        } else {
            seqlen_new = seqlen_k_new_static;
        }
        
        // 计算有效的K序列长度：原长度 - 左填充 + 新增长度
        // if (threadIdx.x == 0) { printf("cur_cu_seqlen_new=%d, next_cu_seqlen_new=%d, seqlen = %d, seqlen_new = %d, leftpad_k = %d\n", cur_cu_seqlen_new, next_cu_seqlen_new, seqlen, seqlen_new, leftpad_k); }
        seqlen = seqlen - leftpad_k + seqlen_new;
        
        // 计算需要的N维度块数（向上取整）
        // ceil(8446 / 64) = 132
        return batch_idx < num_batch && lane < kNumBatchPerWarp
            ? blockn_divmod.div(seqlen + blockn_divmod.divisor - 1) : 0;
    };

    // 计算当前warp处理的批次起始索引
    int warp_idx = threadIdx.x / cutlass::NumThreadsPerWarp; 
    int bidb_start = kNumBatchPerWarp * warp_idx; // 每个批次处理31个
    
    // 获取当前线程负责的批次的M和N块数
    int num_m_blocks = get_num_m_blocks(bidb_start); // 1
    int num_n_blocks = get_num_n_blocks(bidb_start); // 132

    // 计算总的注意力块数（M块数 × N块数）
    int total_blocks = num_m_blocks * num_n_blocks; // 1 * 132 = 132
    
    // Warp内求和：使用shuffle指令在warp内进行归约求和
    #pragma unroll
    for (int i = cutlass::NumThreadsPerWarp / 2; i >= 1; i /= 2) {
        total_blocks += __shfl_down_sync(0xffffffff, total_blocks, i);
    }
    
    // warp内的第一个线程将结果加到共享内存中
    if (lane == 0) { atomicAdd(total_blocks_smem, total_blocks); }
    __syncthreads();
    
    // 所有线程读取总的块数
    total_blocks = total_blocks_smem[0]; // (128~132) * 128
    
    // 计算每个SM需要处理的块数，添加10%的安全边际
    // 130 * 128 * 1.1 * 1 / 132 = 128 * 1.1
    int blocks_per_sm = static_cast<int>(ceilf(float(total_blocks) * 1.1f * float(num_head) / float(num_sm)));
    
    // blocks_per_sm = std::max(1, blocks_per_sm);  // SM处理的最小块数为1
    
    // 计算动态分割数：
    // 1. 理想情况下，分割数 = ceil(N块数 / 每SM的块数)
    // 2. 但不能超过静态分割数的限制
    // 3. 最小为1（不分割）
    // 又除以了，这样翻倍好像也没关系，大概得到1，可能decode模式下num_m_blocks为1，所以total_blocks = num_n_blocks
    // num_splits_dynamic = (num_n_blocks * num_sm/ (total_blocks * 1.1)) = (num_sm / (num_m_blocks * 1.1)) = (num_sm / (batch_size * 1.1)) 看起来和batch有关系
    int num_splits_dynamic = std::max(std::min((num_n_blocks + blocks_per_sm - 1) / blocks_per_sm, num_splits_static), 1);
    
    // 将计算出的动态分割数写入输出数组
    if (bidb_start + lane < num_batch && lane < kNumBatchPerWarp) {
        num_splits_dynamic_ptr[bidb_start + lane] = num_splits_dynamic;
        // printf("idx = %d, num_m_blocks = %d, num_n_blocks = %d, num_split_static = %d, num_splits_dynamic = %d\n", bidb_start + lane, num_m_blocks_ptr[bidb_start + lane], num_n_blocks, num_splits_static, num_splits_dynamic);
    }
}

} // flash

void prepare_varlen_num_blocks(Flash_fwd_params &params, cudaStream_t stream, bool packgqa,
                               int blockM, int blockN, bool enable_pdl) {
    // Only support batch <= 992 (32 warps, each with 31 batches)
    int qhead_per_khead = !packgqa ? 1 : cutlass::ceil_div(params.h, params.h_k); // 16
    flash::prepare_varlen_num_blocks_kernel<<<1 /*grid*/, 1024 /*block*/, 0, stream>>>(
        params.seqlen_q, params.seqlen_k, params.seqlen_knew,
        params.cu_seqlens_q, params.cu_seqlens_k, params.cu_seqlens_knew,
        params.seqused_q, params.seqused_k, params.leftpad_k,
        params.b, !packgqa ? params.h : params.h_k, qhead_per_khead, params.num_sm, params.num_splits,
        cutlass::FastDivmod(blockM), cutlass::FastDivmod(blockN),
        params.tile_count_semaphore,
        // params.num_m_blocks_ptr,
        params.num_splits_dynamic_ptr, enable_pdl);
}

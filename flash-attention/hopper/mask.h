/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

#include "utils.h"

namespace flash {

using namespace cute;

/**
 * Flash Attention Mask类 - 实现各种attention mask机制
 * 
 * 模板参数：
 * - kBlockM, kBlockN: tile的行列大小
 * - PackGQA: 是否使用Packed Group Query Attention优化
 * - TiledMma: 矩阵乘法tile配置
 * - SwapAB: 是否交换A和B矩阵的位置（用于反向传播）
 */
// 64, 64, true, TiledMmaQK, false
template <int kBlockM, int kBlockN, bool PackGQA, typename TiledMma, bool SwapAB=false>
struct Mask {

    static_assert(!(PackGQA && SwapAB), "Cannot be both PackGQA and SwapAB");

    // === 成员变量：mask计算所需的各种参数 ===
    int const thread_idx;                              // 当前线程索引
    int const seqlen_q, seqlen_k;                     // Q和K序列的长度
    int const window_size_left, window_size_right;    // 局部attention的左右窗口大小
    int const sink_token_length;                      // sink token长度（可以全局访问的特殊token数量）
    cutlass::FastDivmod const attention_chunk_divmod; // attention chunk的除法器（用于chunked attention）
    cutlass::FastDivmod const qhead_per_khead_divmod; // Q head到K head的映射除法器（用于GQA）
    cutlass::FastDivmod const qhead_per_khead_mtp_divmod;  // MTP size的除法器

    /**
     * 构造函数：初始化mask计算所需的所有参数
     */
    CUTLASS_DEVICE
    Mask(const int thread_idx, const int seqlen_q, const int seqlen_k,
         const int window_size_left, const int window_size_right, const int sink_token_length,
         cutlass::FastDivmod const &attention_chunk_divmod,
         cutlass::FastDivmod const &qhead_per_khead_divmod,
         cutlass::FastDivmod const &qhead_per_khead_mtp_divmod = cutlass::FastDivmod())
        : thread_idx(thread_idx)
        , seqlen_q(seqlen_q)
        , seqlen_k(seqlen_k)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , sink_token_length(sink_token_length)
        , attention_chunk_divmod(attention_chunk_divmod)
        , qhead_per_khead_divmod(qhead_per_khead_divmod)
        , qhead_per_khead_mtp_divmod(qhead_per_khead_mtp_divmod.divisor == 1 ? qhead_per_khead_divmod : qhead_per_khead_mtp_divmod)
    {
    };

    /**
     * 核心mask应用函数：对注意力分数矩阵应用各种mask
     * 
     * 模板参数：
     * - Seqlenk_mask: 是否应用序列长度mask（处理padding）
     * - Causal_mask: 是否应用因果mask（decoder模式，只能看到之前的token）
     * - Local_mask: 是否应用局部attention mask（限制attention window）
     * 
     * 参数：
     * - tSrS: 注意力分数矩阵（Q*K^T的结果）
     * - m_block, n_block: 当前处理的block索引
     */
    template <bool Seqlenk_mask=false, bool Causal_mask=false, bool Local_mask=false,
        typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block) const {
        // 编译时检查：不能同时是因果mask和局部mask
        static_assert(!(Causal_mask && Local_mask), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        // 如果没有任何mask需要应用，直接返回
        if (!Seqlenk_mask && !Causal_mask && !Local_mask) { return; }

        // === 设置线程和坐标系统 ===
        // thread_idx 为 threadIdx.x - MmaThreadOffset = threadIdx.x - 128
        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);  // 当前线程的MMA slice
        auto thread0_mma = TiledMma{}.get_thread_slice(_0{});       // 线程0的MMA slice（用于获取编译时已知的坐标）

        // 根据SwapAB确定行列维度（SwapAB主要用于反向传播）
        static constexpr int Row = !SwapAB ? 0 : 1, Col = !SwapAB ? 1 : 0;

        // === 创建坐标张量和布局转换 ===
        // 创建单位张量，用于获取每个元素在全局矩阵中的坐标
        Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);  // 当前线程负责的坐标
        
        // 添加调试信息 - 只在thread 0和几个特定线程打印
        // if (thread_idx < 4) {  // 只打印前4个线程
        //     printf("Thread %d: tScS shape: ", thread_idx);
        //     // ((2, 2, 8), 1, 1)
        //     print(tScS.shape()); 
        //     printf(", tScS layout: ");
        //     // ((2, 2, 8), 1, 1):((1@1, 8@0, 8A1), 0, 0)
        //     print(tScS.layout());
        //     printf("\n");
        // }
        
        // 将注意力分数矩阵和坐标张量转换为行列布局，便于mask操作
        // tSrS.layout() (2, 2, 8)
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));
        
        // 只在thread 0打印布局转换信息
        // if (thread_idx == 0) {
        //     printf("Thread 0 Layout Conversion:\n");
        //     printf("  Original tSrS layout: ");
        //     // ((_2,_2,_8),_1,_1):((_1,_2,_4),_0,_0)
        //     print(tSrS.layout());
        //     printf("\n  Converted tSrS_rowcol layout: ");
        //     // ((_2,_1),(_2,_8,_1)):((_2,_0),(_1,_4,_0))
        //     print(tSrS_rowcol.layout());
        //     printf("\n  Original tScS layout: ");
        //     // ((_2,_2,_8),_1,_1):((_1@1,_8@0,_8@1),_0,_0)
        //     print(tScS.layout()); 
        //     printf("\n  Converted tScS_rowcol layout: ");
        //     // ((_2,_1),(_2,_8,_1)):((_8@0,_0),(_1@1,_8@1,_0))
        //     print(tScS_rowcol.layout());
        //     printf("\n");
        // }
        
        Tensor t0ScS = thread0_mma.partition_C(cS);  // 线程0的坐标（编译时已知）
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(t0ScS.layout()));
        
        // 比较当前线程和线程0的坐标差异
        // if (thread_idx == 0) {
        //     printf("Thread 0 coordinates (first 8 elements): ");
        //     for (int i = 0; i < min(8, size(t0ScS_rowcol)); ++i) {
        //         auto coord = t0ScS_rowcol(i);
        //         // (0,0) (8,0) (0,1) (8,1) (0,8) (8,8) (0,9) (8,9) 
        //         printf("(%d,%d) ", int(get<0>(coord)), int(get<1>(coord)));
        //     }
        //     printf("\n");
        // }
        // if (thread_idx == 1) {  // 也打印thread 1作为对比
        //     printf("Thread 1 coordinates (first 8 elements): ");
        //     for (int i = 0; i < min(8, size(tScS_rowcol)); ++i) {
        //         auto coord = tScS_rowcol(i);
        //         // (0,2) (8,2) (0,3) (8,3) (0,10) (8,10) (0,11) (8,11) 
        //         printf("(%d,%d) ", int(get<0>(coord)), int(get<1>(coord)));
        //     }
        //     printf("\n");
        // }
        
        // === 计算mask边界 ===
        // 使用线程0的列坐标进行比较，因为它在编译时已知，可以优化性能
        // 减去当前线程的列偏移，得到相对于线程0的列限制
        int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
        // 减去 thread_col_offset 是为了获得相对于这个 64 * 64 块的列限制
        int const seqlenk_col_limit = seqlen_k - n_block * kBlockN - thread_col_offset - qhead_per_khead_mtp_divmod.divide(qhead_per_khead_divmod.divisor) + 1;
        
        // === 应用不同类型的mask ===
        if constexpr (!Causal_mask && !Local_mask) {
            // === 情况1：只需要序列长度mask ===
            if constexpr (Seqlenk_mask) {
                // 遍历当前线程负责的所有行（Query tokens）
                #pragma unroll
                for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                    // 获取当前行索引
                    int const row_idx = get<Row>(tScS_rowcol(m, _0{}));
                    int const mtp_index = qhead_per_khead_mtp_divmod.divide(row_idx);
                    // 根据行索引判断是否需要减1
                    int const seqlenk_col_limit_mtp = seqlenk_col_limit + mtp_index;
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= seqlenk_col_limit_mtp) {
                            tSrS_rowcol(m, n) = -INFINITY;  // 设置为负无穷，softmax后变为0
                        }
                    }
                }
            }
        } else {
            // === 情况2：需要基于行列位置的复杂mask（因果mask或局部mask） ===
            if constexpr (!SwapAB) {
                // === 前向传播模式的mask ===
                
                // === PackGQA模式下的头映射计算 ===
                // PackGQA (Packed Grouped Query Attention) 需要将多个Query头映射到同一个Key头
                // 为了避免重复的divmod计算，同一行的线程共享计算结果
                // AtomLayoutC_TV Shape((4, 8, 4), (2, 2, 8)), Stride((128, 1, 16), (64, 8, 512))
                static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
                static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
                static_assert(!PackGQA || CUTE_STATIC_V(size<0>(tSrS_rowcol)) <= kMmaThreadsPerRow);
                int mma_m_idx;
                
                // PackGQA模式下计算Q head到K head的映射
                if constexpr (PackGQA) {
                    // 计算当前Query块和线程对应的Key头索引
                    // 使用divmod将Query头索引映射到Key头索引（多对一映射）
                    // 看起来是用thread_idx % kMmaThreadsPerRow来获取当前线程负责第几行，然后divide khead来获取当前线程负责的
                    mma_m_idx = qhead_per_khead_divmod.divide(m_block * kBlockM + get<Row>(tScS_rowcol(thread_idx % kMmaThreadsPerRow, _0{})));
                }
                
                // === 因果mask偏移量计算 ===
                // 计算因果mask的行偏移：确保Q的第i个token只能看到K的前i个token
                // 这个偏移量定义了当前Query token能看到的Key token范围的右边界
                //
                // 计算公式解析：
                // - seqlen_k: Key序列的总长度
                // - n_block * kBlockN: 当前Key块的起始位置（全局坐标）
                // - seqlen_q: Query序列长度（通常等于seqlen_k，但在推理时可能不同）
                // - thread_col_offset: 当前线程负责的列范围的起始偏移
                // 
                // 最终结果：对于位置i的Query token，它最多能看到位置i的Key token
                int const causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q - thread_col_offset;
                
                if constexpr (Causal_mask) {
                    // === 因果mask：实现decoder的自回归特性 ===
                    // 遍历当前线程负责的所有行（Query tokens）
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        // === 获取当前行的全局索引 ===
                        int const row_idx = !PackGQA
                            // 标准模式：块内行索引 + 块偏移 = 全局行索引
                            ? get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                            // PackGQA模式：使用warp内shuffle操作获取共享的头映射结果
                            // 这避免了每个线程重复计算divmod，提高效率
                            : __shfl_sync(0xffffffff,                    // 参与的线程掩码（所有线程）
                                         mma_m_idx,                      // 要传递的数据（头映射索引）
                                         m % kMmaThreadsPerRow,          // 源线程ID
                                         kMmaThreadsPerRow);             // 线程组大小
                        
                        // === 计算右边界：当前token只能看到它之前的token ===
                        // row_idx: 当前Query token的位置
                        // causal_row_offset: 偏移量，确定可见范围
                        // seqlenk_col_limit: Key序列长度限制
                        int const col_limit_right = !Seqlenk_mask
                            ? row_idx + causal_row_offset  // 简单情况：位置i的token只能看到位置0到i
                            // 复杂情况：同时考虑因果mask和序列长度mask
                            // __viaddmin_s32(a, b, c) = min(a + b, c)
                            // 这是CUDA的SIMD向量化内置函数，高效计算 min(row_idx + causal_row_offset, seqlenk_col_limit)
                            // 确保mask边界不超过实际序列长度
                            : __viaddmin_s32(row_idx, causal_row_offset, seqlenk_col_limit);
                        
                        // === 应用因果mask ===
                        // 遍历当前行的所有列（Key tokens）
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            // 如果Key token的位置超出了当前Query token能看到的范围
                            // 将注意力分数设置为负无穷，softmax后会变成0
                            if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= col_limit_right) { 
                                tSrS_rowcol(m, n) = -INFINITY;  // mask掉不能看到的位置
                            }
                        }
                    }
                } else {
                    // === 局部attention mask：限制attention window大小 ===
                    int const local_row_offset_right = causal_row_offset + window_size_right;  // 右窗口边界
                    int const local_row_offset_left = causal_row_offset - 1 - window_size_left; // 左窗口边界
                    int const col_limit_sink = sink_token_length - n_block * kBlockN;  // sink token边界
                    
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        // 获取当前行索引
                        int const row_idx = !PackGQA
                            ? get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                            : __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                        
                        // 计算局部attention的左右边界
                        int col_limit_right = !Seqlenk_mask
                            ? row_idx + local_row_offset_right
                            : __viaddmin_s32(row_idx, local_row_offset_right, seqlenk_col_limit);
                        int col_limit_left = row_idx + local_row_offset_left;
                        
                        // 如果使用chunked attention，进一步限制边界
                        if (attention_chunk_divmod.divisor > 0) {
                            int col_limit_left_chunk = flash::round_down(attention_chunk_divmod, row_idx + seqlen_k - seqlen_q) - n_block * kBlockN - thread_col_offset;
                            col_limit_left = std::max(col_limit_left, col_limit_left_chunk);
                            col_limit_right = std::min(col_limit_right, col_limit_left_chunk + attention_chunk_divmod.divisor);
                        }
                        
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            int const col_idx = int(get<Col>(t0ScS_rowcol(m, n)));
                            // mask掉窗口外的位置，但保留sink token
                            if (col_idx >= col_limit_right || (col_idx < col_limit_left && col_idx >= col_limit_sink)) { 
                                tSrS_rowcol(m, n) = -INFINITY; 
                            }
                        }
                    }
                }
            } else {
                // === 反向传播模式的mask（SwapAB=true） ===
                // TODO: 反向传播暂不支持attention_chunk
                int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
                int const causal_row_offset = seqlenk_col_limit - seqlen_q + m_block * kBlockM + thread_row_offset;
                
                if constexpr (Causal_mask) {
                    // === 反向传播的因果mask ===
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        // 如果列超出边界，mask整列；否则根据因果关系确定行边界
                        int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset;
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                            if (int(get<Row>(t0ScS_rowcol(m, _0{}))) < row_limit_top) { 
                                tSrS_rowcol(m, n) = -INFINITY; 
                            }
                        }
                    }
                } else {
                    // === 反向传播的局部attention mask ===
                    int const col_limit_sink = sink_token_length - n_block * kBlockN - thread_col_offset;
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        // 计算局部窗口的上下边界
                        int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset - window_size_right;
                        int const row_limit_bot = col0 < col_limit_sink ? kBlockM : col0 - causal_row_offset + window_size_left;
                        
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                            int const row_idx = int(get<Row>(t0ScS_rowcol(m, _0{})));
                            // mask掉窗口外的位置
                            if (row_idx < row_limit_top || row_idx > row_limit_bot) { 
                                tSrS_rowcol(m, n) = -INFINITY; 
                            }
                        }
                    }
                }
            }
        }
    };

};

} // namespace flash

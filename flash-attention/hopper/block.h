/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

namespace flash {

/**
 * BlockMN模板类：用于Flash Attention中计算块(block)的范围和索引
 * 
 * 模板参数：
 * - SeqlenInfo_t: 序列长度信息的类型
 * - kBlockM: M维度(查询序列)的块大小
 * - kBlockN: N维度(键值序列)的块大小  
 * - Is_causal: 是否启用因果掩码(causal mask)，用于解码器自注意力
 * - Is_local: 是否启用局部注意力(local attention)，限制注意力窗口大小
 * - PackGQA: 是否启用分组查询注意力(Grouped Query Attention)打包优化
 * - Split: 是否启用分割计算，用于并行处理大序列
 */
template <class SeqlenInfo_t, int kBlockM, int kBlockN, bool Is_causal, bool Is_local, bool PackGQA=false, bool Split=false>
struct BlockMN {

    /**
     * 计算N维度(键值序列维度)上需要处理的块范围[n_block_min, n_block_max)
     * 
     * 参数：
     * - seqlen_info: 包含序列长度信息的结构体
     * - m_block: 当前处理的M维度块索引
     * - bidb: 批次中的序列索引  
     * - split_idx: 分割索引，用于并行处理
     * - num_splits: 总分割数量
     * - window_size_left/right: 局部注意力的左右窗口大小
     * - attention_chunk_divmod: 注意力块分割的除法模运算器
     * - qhead_per_khead_divmod: 查询头与键值头比例的除法模运算器
     * 
     * 返回：tuple<int, int> 表示 [n_block_min, n_block_max)
     */
    static
    CUTLASS_DEVICE
    cute::tuple<int, int> get_n_block_min_max(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const bidb, int const split_idx, int const num_splits,
            int const window_size_left, int const window_size_right,
            cutlass::FastDivmod const& attention_chunk_divmod,
            cutlass::FastDivmod const& qhead_per_khead_divmod) {

        int const seqlen_k = seqlen_info.seqlen_k;  // 键值序列长度
        int const seqlen_q = seqlen_info.seqlen_q;  // 查询序列长度
        
        // 计算N维度的最大块数量（基于键值序列长度）
        int n_block_max = cute::ceil_div(seqlen_k, kBlockN);
        
        // 如果启用因果掩码或局部注意力，需要限制n_block_max
        if constexpr (Is_causal || Is_local) {
            // 计算当前M块对应的最大查询索引
            int m_idx_max = (m_block + 1) * kBlockM;
            
            // TODO: 检查off-by-1错误
            // 如果启用GQA打包，需要根据查询头与键值头的比例调整索引
            if (PackGQA) { 
                m_idx_max = qhead_per_khead_divmod.divide(m_idx_max - 1) + 1; 
            }
            
            // 计算对应的N维度索引，考虑序列长度差异
            int const n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q;
            
            // 根据是否局部注意力调整右边界
            int n_idx_right = !Is_local ? n_idx : n_idx + window_size_right;
            
            // 如果启用局部注意力且有注意力块分割，进一步限制右边界
            if (Is_local && attention_chunk_divmod.divisor > 0) {
                n_idx_right = std::min(n_idx_right, flash::round_up(attention_chunk_divmod, n_idx));
            }
            
            // 更新n_block_max为计算出的右边界对应的块数
            n_block_max = std::min(n_block_max, cute::ceil_div(n_idx_right, kBlockN));
        }
        
        // 计算N维度的最小块索引
        int n_block_min = 0;
        if constexpr (Is_local) {
            // 计算当前M块对应的最小查询索引
            int m_idx_min = m_block * kBlockM;
            if (PackGQA) { 
                m_idx_min = qhead_per_khead_divmod.divide(m_idx_min); 
            }
            
            // 计算对应的N维度索引
            int const n_idx = m_idx_min + seqlen_k - seqlen_q;
            
            // 计算局部注意力的左边界
            int n_idx_left = n_idx - window_size_left;
            
            // 如果有注意力块分割，调整左边界
            if (attention_chunk_divmod.divisor > 0) {
                n_idx_left = std::max(n_idx_left, flash::round_down(attention_chunk_divmod, n_idx));
            }
            
            // 计算对应的最小块索引，确保不小于0
            n_block_min = std::max(int(0), n_idx_left / kBlockN);
        }
        
        // 调试输出（注释掉的代码）
        // if (threadIdx.x == 128) { printf("Inside, bid.x = %d, bid.y = %d, bid.z = %d, split_idx = %d, n_block_min: %d, n_block_max: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, split_idx, n_block_min, n_block_max); }
        
        // 如果启用分割计算，进一步细分块范围
        if constexpr (Split) {
            // 从split_idx的高16位提取动态分割数量
            uint32_t num_splits_dynamic_u = reinterpret_cast<uint32_t const&>(split_idx) >> 16;
            int num_splits_dynamic = reinterpret_cast<int&>(num_splits_dynamic_u);
            
            // 从split_idx的低16位提取实际分割索引
            int split_idx_actual = split_idx & 0x0000FFFF;
            
            // 确定实际使用的分割数量
            int num_splits_actual = num_splits_dynamic > 0 ? num_splits_dynamic : num_splits;
            
            // 计算每个分割应处理的块数量 132
            int num_n_blocks_per_split = n_block_max <= n_block_min ? 0 : cute::ceil_div(n_block_max - n_block_min, num_splits_actual);
            
            // 根据当前分割索引调整块范围 0 + 0 = 0
            n_block_min = n_block_min + split_idx_actual * num_n_blocks_per_split;
            n_block_max = std::min(n_block_min + num_n_blocks_per_split, n_block_max); // 132
            
            // 调试输出（注释掉的代码）
            // if (threadIdx.x == 128) { printf("Inside, bid.x = %d, bid.y = %d, bid.z = %d, split_idx = %d, num_splits_dynamic = %d, num_splits_actual = %d, num_n_blocks_per_split = %d, n_block_min: %d, n_block_max: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, split_idx, num_splits_dynamic, num_splits_actual, num_n_blocks_per_split, n_block_min, n_block_max); }
        }
        
        // 调试输出（注释掉的代码）
        // if (threadIdx.x == 128) { printf("After split, inside, bid.y = %d, bid.z = %d, split_idx = %d, n_block_min: %d, n_block_max: %d\n", blockIdx.y, blockIdx.z, split_idx, n_block_min, n_block_max); }
        
        return {n_block_min, n_block_max};
    }

    /**
     * 计算新键值序列(K_new)对应的N维度块范围
     * 用于处理键值缓存更新场景，其中需要区分原有键值(K_og)和新添加的键值(K_new)
     * 
     * 返回：tuple<int, int> 表示新键值序列的 [n_block_new_min, n_block_new_max)
     */
    static
    CUTLASS_DEVICE
    cute::tuple<int, int> get_n_block_k_new_min_max(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const bidb, int const split_idx, int const num_splits,
            int const window_size_left, int const window_size_right,
            cutlass::FastDivmod const& attention_chunk_divmod,
            cutlass::FastDivmod const& qhead_per_khead_divmod) {

        // 首先获取完整的N维度块范围
        auto [n_block_min, n_block_max] = get_n_block_min_max(
            seqlen_info, m_block, bidb, split_idx, num_splits,
            window_size_left, window_size_right, attention_chunk_divmod, qhead_per_khead_divmod);
            
        // 计算新键值序列在完整序列中的索引范围
        int const idx_k_new_min = std::max(n_block_min * kBlockN - seqlen_info.seqlen_k_og, 0);
        int const idx_k_new_max = std::min(n_block_max * kBlockN - seqlen_info.seqlen_k_og, seqlen_info.seqlen_k_new);
        
        // 将索引范围转换为块范围
        int const n_block_new_min = idx_k_new_min / kBlockN;
        int const n_block_new_max = idx_k_new_max > idx_k_new_min ? cute::ceil_div(idx_k_new_max, kBlockN) : n_block_new_min;
        
        // 调试输出（注释掉的代码）
        // if (threadIdx.x == 128 && m_block == 0) { printf("bidb = %d, seqlen_k_new = %d, seqlen_k_og = %d, n_block_min = %d, n_block_max = %d, idx_k_new_min = %d, idx_k_new_max = %d, n_block_new_min = %d, n_block_new_max = %d\n", bidb, seqlen_k_new, seqlen_k_og, n_block_min, n_block_max, idx_k_new_min, idx_k_new_max, n_block_new_min, n_block_new_max);}
        
        return {n_block_new_min, n_block_new_max};
    }

    /**
     * 计算M维度(查询序列维度)上需要处理的块范围[m_block_min, m_block_max)
     * 用于反向计算：给定N维度的块索引，确定哪些M维度的块需要处理
     * 
     * 参数：
     * - n_block: 当前处理的N维度块索引
     * - sink_token_length: sink token的长度，用于特殊的注意力模式
     * 
     * 返回：tuple<int, int> 表示 [m_block_min, m_block_max)
     */
    static
    CUTLASS_DEVICE
    cute::tuple<int, int> get_m_block_min_max(
            SeqlenInfo_t const& seqlen_info,
            int const n_block, int const bidb,
            int const window_size_left, int const window_size_right, int const sink_token_length) {
        // TODO: 支持attention_chunk
        int const seqlen_q = seqlen_info.seqlen_q;  // 查询序列长度
        int const seqlen_k = seqlen_info.seqlen_k;  // 键值序列长度
        
        // 计算M维度的最大块数量（基于查询序列长度）
        int m_block_max = cute::ceil_div(seqlen_q, kBlockM);
        
        // 如果启用局部注意力，需要考虑窗口限制
        if constexpr (Is_local) {
            // 对于超出sink token范围的块，需要限制M维度的范围
            if (n_block >= cute::ceil_div(sink_token_length, kBlockN)) {
                // 根据局部注意力窗口计算M维度的上界
                m_block_max = std::min(m_block_max, 
                    cute::ceil_div((n_block + 1) * kBlockN + seqlen_q - seqlen_k + window_size_left, kBlockM));
            }
        }
        
        // 计算M维度的最小块索引
        int m_block_min = 0;
        if constexpr (Is_causal || Is_local) {
            // 根据因果掩码或局部注意力窗口计算M维度的下界
            m_block_min = std::max(m_block_min, 
                (n_block * kBlockN + seqlen_q - seqlen_k - window_size_right) / kBlockM);
        }
        
        return {m_block_min, m_block_max};
    }

    /**
     * 计算在因果掩码或局部掩码情况下，N维度块的最小索引
     * 用于确定需要特殊处理（应用掩码）的迭代范围的起始点
     * 
     * 返回：int 表示需要开始应用因果/局部掩码的N维度块索引
     */
    static
    CUTLASS_DEVICE
    int get_n_block_min_causal_local_mask(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const n_block_min, int const window_size_right,
            cutlass::FastDivmod const& attention_chunk_divmod,
            cutlass::FastDivmod const& qhead_per_khead_divmod) {
        // 计算当前M块对应的最小查询索引
        int const m_idx_min = !PackGQA ? m_block * kBlockM : qhead_per_khead_divmod.divide(m_block * kBlockM);
        
        // 计算对应的N维度索引
        int const n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q;
        
        // 根据是否局部注意力调整右边界
        int n_idx_right = !Is_local ? n_idx : n_idx + window_size_right;
        
        // 如果启用局部注意力且有注意力块分割，进一步调整边界
        if (Is_local && attention_chunk_divmod.divisor > 0) {
            n_idx_right = std::min(n_idx_right, flash::round_up(attention_chunk_divmod, n_idx));
        }
        
        // 返回需要开始应用掩码的块索引
        return std::max(n_block_min, n_idx_right / kBlockN);
    }

    /**
     * 计算局部掩码之前的N维度块范围
     * 用于确定可以进行常规（无掩码）迭代的范围，在此范围内不需要应用局部掩码
     * 
     * 返回：int 表示常规迭代结束、开始应用局部掩码的N维度块索引
     */
    static
    CUTLASS_DEVICE
    int get_n_block_min_before_local_mask(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const n_block_min, int const window_size_left,
            cutlass::FastDivmod const& attention_chunk_divmod,
            cutlass::FastDivmod const& qhead_per_khead_divmod) {
        // 计算当前M块对应的最大查询索引
        int const m_idx_max = !PackGQA ? (m_block + 1) * kBlockM : qhead_per_khead_divmod.divide((m_block + 1) * kBlockM - 1) + 1;
        
        // 计算对应的N维度索引
        int const n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q;
        
        // 根据是否局部注意力调整左边界
        int n_idx_left = !Is_local ? n_idx : n_idx - window_size_left;
        
        // 如果启用局部注意力且有注意力块分割，进一步调整边界
        if (Is_local && attention_chunk_divmod.divisor > 0) {
            n_idx_left = std::max(n_idx_left, flash::round_down(attention_chunk_divmod, n_idx));
        }
        
        // 返回常规迭代的结束点（局部掩码的开始点）
        return !Is_local ? n_block_min : std::max(n_block_min, cute::ceil_div(n_idx_left, kBlockN));
    }

};

} // namespace flash

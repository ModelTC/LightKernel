/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * 线程级别的归约操作
 * @param tensor 输入张量（2D）
 * @param summary 归约结果张量（1D）
 * @param op 归约操作符（如求最大值、求和等）
 * @param zero_init 是否在第一次迭代时初始化为零
 */
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ni++) {
        #pragma unroll
        for (int mi = 0; mi < size<0>(tensor); mi++) {
            // 如果是第一次迭代且zero_init为true，则直接赋值，否则执行归约操作
            summary(mi) = zero_init && ni == 0 ? tensor(mi, ni) : op(summary(mi), tensor(mi, ni));
        }
    }
}

/**
 * 在4个线程之间进行全归约操作（用于warp内通信）
 * @param dst 目标张量
 * @param src 源张量
 * @param op 归约操作符
 */
template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++) {
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

/**
 * 完整的归约操作：先进行线程级归约，再进行线程间全归约
 */
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

/**
 * 计算张量每行的最大值
 */
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

/**
 * 计算张量每行的和
 * @param warp_reduce 是否进行warp级别的归约
 */
template<bool zero_init=true, bool warp_reduce=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
    if constexpr (warp_reduce) { quad_allreduce_(sum, sum, sum_op); }
}

/**
 * 对张量应用缩放和指数函数
 * @param Scale_max 是否对最大值进行缩放
 * @param Check_inf 是否检查无穷大值
 * @param Max_offset 最大值偏移量（用于FP8优化）
 * @param tensor 输入张量
 * @param max 每行的最大值
 * @param scale 缩放因子
 */
template <bool Scale_max=true, bool Check_inf=true, int Max_offset=0,
        typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    // 对于FP8，我们可以将max减去8.0，使得exp2后的值在[0, 256]范围内
    // 这让我们能够使用更多的FP8范围（而不是仅仅[0, 1]）来减少下溢
    static constexpr float max_offset = float(Max_offset);  // 只能在int上模板化，不能在float上
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // 如果max是-inf，那么所有元素都必须是-inf（可能由于掩码）
        // 我们不希望(-inf - (-inf))，因为那会产生NaN
        const float max_scaled = Check_inf
            ? (max(mi) == -INFINITY ? 0.f : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset)
            : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // 不是计算exp(x - max)，而是计算exp2(x * log_2(e) - max * log_2(e))
            // 这允许编译器使用ffma指令，而不是分别使用fadd和fmul
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * Flash Attention中的在线Softmax实现
 * 使用数值稳定的在线算法来计算softmax，避免存储完整的注意力矩阵
 * 
 * @tparam kNRows 处理的行数
 * @tparam Max_offset 最大值偏移量，用于FP8等低精度优化
 */
// <2, 0>
template <int kNRows, int Max_offset=0>
struct Softmax {

    // 定义张量类型，用于存储每行的统计信息
    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;  // 每行的最大值和累积和
    float const softmax_scale_log2;  // softmax缩放因子的log2值

    /**
     * 构造函数
     * @param softmax_scale_log2_ softmax缩放因子的log2值（通常是1/sqrt(d_k)的log2）
     */
    CUTLASS_DEVICE Softmax(float const softmax_scale_log2_) : softmax_scale_log2(softmax_scale_log2_) {};

    /**
     * 计算最大值并获取缩放因子（在线softmax的核心步骤）
     * 这是实现数值稳定softmax的关键函数，用于处理新的分数块
     * 
     * @tparam Is_first 是否是第一个分数块
     * @tparam Check_inf 是否检查无穷大值
     * @param acc_s 累积的注意力分数张量
     * @return 返回用于重新缩放先前输出的缩放因子
     */
    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    __forceinline__ __device__ TensorT max_get_scale(Tensor0 &acc_s) {
        // 将acc_s从((2, 2, V), MMA_M, MMA_N)重塑为(nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        TensorT scores_scale;
        
        if constexpr (Is_first) {
            // 第一个块：直接计算最大值，缩放因子设为1
            // 某些行全为-inf, row_max 为 -inf
            flash::template reduce_max</*zero_init=*/true>(scores, row_max);
            cute::fill(scores_scale, 1.f);
        } else {
            // 后续块：需要更新最大值并计算重新缩放因子
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);  // 保存之前的最大值 -inf
            flash::template reduce_max</*zero_init=*/false>(scores, row_max);  // 更新全局最大值
            
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                // 处理无穷大的情况
                // 
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                
                // 计算重新缩放因子：exp((old_max - new_max) * scale)
                // 这用于调整之前计算的softmax值
                scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                
                // 使用缩放因子更新累积和
                row_sum(mi) *= scores_scale(mi);
            }
        }
        return scores_scale;
    };

    /**
     * 在线softmax计算
     * 对当前分数块应用softmax并累积到row_sum中
     * 
     * @tparam Is_first 是否是第一个分数块
     * @tparam Check_inf 是否检查无穷大值
     * @param acc_s 累积的注意力分数张量
     */
    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    __forceinline__ __device__ void online_softmax(Tensor0 &acc_s) {
        // 将acc_s从((2, 2, V), MMA_M, MMA_N)重塑为(nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        
        // 应用缩放和指数函数：scores = exp2((scores - max) * scale)
        flash::template scale_apply_exp2</*Scale_max=*/true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2);
        
        // 我们这里不进行跨线程的归约，因为我们不需要立即使用row_sum
        // 我们在最后需要归一化softmax时才进行归约
        flash::reduce_sum</*zero_init=*/Is_first, /*warp_reduce=*/false>(scores, row_sum);
    };

    /**
     * 完成softmax计算并返回最终的缩放因子
     * 这是在线softmax的最后一步，计算1/sum并准备最终的归一化
     * 
     * @param final_scale 最终的缩放因子（默认为1.0）
     * @return 返回用于归一化输出的缩放因子
     */
    __forceinline__ __device__ TensorT finalize(float const final_scale=1.f) {
        SumOp<float> sum_op;
        // 在4个线程之间进行全归约，获取完整的行和
        quad_allreduce_(row_sum, row_sum, sum_op);
        TensorT scores_scale;
        
        #pragma unroll
        for (int mi = 0; mi < size(row_sum); ++mi) {
            float sum = row_sum(mi);
            // 计算倒数，处理sum为0或NaN的情况
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
            scores_scale(mi) = inv_sum * final_scale;
            
            // 对于FP8，我们可能已经将exp的输出缩放了2**8，所以需要将sum除以该数量
            if constexpr (Max_offset != 0) {
                static constexpr float sum_scale = 1.f / float(1 << Max_offset);
                sum *= sum_scale;
            }
            
            // 更新row_sum为log-sum-exp值（用于后续的梯度计算或其他用途）
            // LSE = max + log(sum)，其中sum是exp(scores - max)的和
            row_sum(mi) = (sum == 0.f || sum != sum) ? -INFINITY : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
        }
        return scores_scale;
    };

    /**
     * 重新缩放输出张量
     * 使用计算得到的缩放因子来调整输出，这在在线softmax中是必要的
     * 
     * @param acc_o 累积的输出张量
     * @param scores_scale 缩放因子张量
     */
    template<typename Tensor1>
    __forceinline__ __device__ void rescale_o(Tensor1 &acc_o, TensorT const &scores_scale) {
        // 将acc_o从(MMA=4, MMA_M, MMA_K)重塑为(nrow=(2, MMA_M), ncol=(2, MMA_K))
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(CUTE_STATIC_V(size<0>(acc_o_rowcol)) == kNRows);
        
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { 
                // 对每个输出元素应用对应行的缩放因子
                acc_o_rowcol(mi, ni) *= scores_scale(mi); 
            }
        }
    };

};

}  // namespace flash

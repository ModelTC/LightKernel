/*
 * =====================================================================
 * Flash Attention 双 Warpgroup 协作架构详细说明
 * =====================================================================
 * 
 * 概述：
 * Flash Attention 在 H100 GPU 上采用创新的双 warpgroup 协作模式，通过
 * 生产者-消费者架构实现计算和内存访问的完美重叠，显著提升性能。
 * 
 * =====================================================================
 * 架构设计
 * =====================================================================
 * 
 * Warpgroup 0 (生产者) - 数据加载管道：
 * ┌─────────────────────────────────────────────────────────────────┐
 * │ 职责：                                                           │
 * │ • 从全局内存异步加载 Q、K、V 张量到共享内存                      │
 * │ • 管理 TMA (Tensor Memory Accelerator) 硬件加速传输             │
 * │ • 处理 PagedKV 页表管理和动态内存分配                           │
 * │ • 执行 V 张量转置操作（如果需要）                                │
 * │ • 协调多级流水线状态，实现高效的内存预取                         │
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * Warpgroup 1 (消费者) - 计算执行管道：
 * ┌─────────────────────────────────────────────────────────────────┐
 * │ 职责：                                                           │
 * │ • 从共享内存读取 Q、K、V 张量数据                               │
 * │ • 执行 Q@K^T 矩阵乘法计算注意力分数                            │
 * │ • 应用 Online Softmax 算法进行归一化                           │
 * │ • 执行 Attention@V 矩阵乘法得到最终输出                        │
 * │ • 处理 FP8 量化、因果掩码、滑动窗口等高级特性                   │
 * │ • 管理输出张量的存储和 LSE 累积                                 │
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * =====================================================================
 * 关键同步机制详解
 * =====================================================================
 * 
 * 1. QueryEmpty Barrier：
 * ┌─────────────────────────────────────────────────────────────────┐
 * │ 目的：确保 Q 共享内存区域的安全访问                              │
 * │ 参与者：                                                         │
 * │   • TMA 模式：NumMmaThreadsQK + NumThreadsPerWarp               │
 * │   • cp.async 模式：NumMmaThreadsQK + NumProducerThreads         │
 * │ 机制：Named Barrier - 双向同步                                  │
 * │ 时序：                                                           │
 * │   ① 消费者完成 Q@K^T 计算后释放 Q 区域                          │
 * │   ② 生产者等待确认，然后安全覆盖 Q 数据                         │
 * │   ③ 生产者开始新一轮 Q 张量加载                                 │
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * 2. barrier_O (输出屏障)：
 * ┌─────────────────────────────────────────────────────────────────┐
 * │ 目的：协调输出完成与 V 区域重用                                  │
 * │ 参与者：生产者和消费者通过 ClusterBarrier 协调                   │
 * │ 机制：ClusterBarrier (跨 CTA 同步，非 NamedBarrier)             │
 * │ 重要性：                                                         │
 * │   • 防止生产者过早覆盖 V 数据                                   │
 * │   • 确保 TMA 多播操作的正确性                                   │
 * │   • 协调集群内不同 CTA 的执行时序                               │
 * │ 时序：                                                           │
 * │   ① 消费者完成 Attention@V 计算和输出写入                       │
 * │   ② 消费者信号 barrier_O                                       │
 * │   ③ 生产者等待确认                                              │
 * │   ④ 生产者开始安全加载新的 V 数据                               │
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * 3. TransposeBarrier：
 * ┌─────────────────────────────────────────────────────────────────┐
 * │ 目的：确保 V 张量转置操作的原子性                                │
 * │ 参与者：NumThreadsPerWarpGroup (单个 warpgroup 所有线程)        │
 * │ 机制：Named Barrier - warpgroup 内同步                          │
 * │ 流程：                                                           │
 * │   ① 等待 Vt 数据在共享内存中可用                                │
 * │   ② 获取 V 流水线写入权限                                       │
 * │   ③ 执行 LDSM/STSM 转置操作                                     │
 * │   ④ 内存栅栏确保转置完成                                        │
 * │   ⑤ TransposeBarrier 同步确保 warpgroup 协调                   │
 * │   ⑥ 释放 Vt 流水线                                             │
 * │ 重要性：PipelineTmaAsync::consumer_release 要求同步             │
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * 4. AppendKV Barrier：
 * ┌─────────────────────────────────────────────────────────────────┐
 * │ 目的：协调增量注意力中的 KV_new 张量处理                         │
 * │ 参与者：NumMmaThreads + NumProducerThreads (所有线程)           │
 * │ 机制：Named Barrier - 双向同步点                                │
 * │ 应用场景：增量生成中新 token 的 K/V 处理                        │
 * │ 流程：                                                           │
 * │   ① 生产者加载 KV_new 到共享内存                                │
 * │   ② AppendKV Barrier 同步确保数据可见                          │
 * │   ③ 消费者读取并存储 KV_new 到全局内存                         │
 * │   ④ fence.proxy.async.global 确保写入可见                     │
 * │   ⑤ 消费者 arrive barrier 完成同步                            │
 * └─────────────────────────────────────────────────────────────────┘
 * 
 * =====================================================================
 * 性能优化策略
 * =====================================================================
 * 
 * 1. 流水线重叠：
 *    • 计算与内存访问完全重叠
 *    • 多阶段流水线状态管理
 *    • 预取和缓存优化
 * 
 * 2. 内存层次优化：
 *    • TMA 硬件加速的全局内存访问
 *    • 共享内存的高效布局和访问模式
 *    • Register 文件的精细化管理
 * 
 * 3. 同步开销最小化：
 *    • 精确的 barrier 设计避免过度同步
 *    • 集群级别的协调优化
 *    • 条件同步减少不必要的等待
 * 
 * 4. 数据重用优化：
 *    • 共享内存区域的动态重用
 *    • 张量布局优化支持高效访问
 *    • 多播机制减少重复传输
 * 
 * =====================================================================
 * 特殊特性支持
 * =====================================================================
 * 
 * • FP8 量化：支持混合精度计算和动态反缩放
 * • 因果注意力：优化的掩码处理和计算跳过
 * • 滑动窗口：局部注意力的高效实现
 * • 变长序列：动态序列长度处理
 * • 分组查询注意力 (GQA)：多查询头共享键值头
 * • PagedKV：动态内存管理和非连续存储
 * • 集群级别优化：多 SM 协作和负载平衡
 * 
 * =====================================================================
 */

/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cutlass/arch/grid_dependency_control.h"

#include "seqlen.h"
#include "utils.h"
#include "softmax.h"

namespace flash {

using namespace cute;

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class FlashAttnFwdSm90 {

public:

    // Type Aliases
    using CollectiveMainloop = CollectiveMainloop_;
    using CollectiveEpilogue = CollectiveEpilogue_;
    static constexpr bool Is_causal = CollectiveMainloop::Is_causal;
    static constexpr bool Is_local = CollectiveMainloop::Is_local;
    static_assert(CollectiveMainloop::Varlen == CollectiveEpilogue::Varlen);
    static constexpr bool Has_softcap = CollectiveMainloop::Has_softcap;
    static constexpr bool Varlen = CollectiveMainloop::Varlen;
    static constexpr bool Split = CollectiveMainloop::Split;
    static constexpr bool Is_FP8 = CollectiveMainloop::Is_FP8;
    static constexpr bool Transpose_V = CollectiveMainloop::Transpose_V;
    static constexpr bool AppendKV = CollectiveMainloop::AppendKV;
    static constexpr bool HasQv = CollectiveMainloop::HasQv;
    static constexpr bool Use_TMA_Q = CollectiveMainloop::Use_TMA_Q;
    static constexpr bool Use_TMA_KV = CollectiveMainloop::Use_TMA_KV;
    static constexpr bool Use_TMA_O = CollectiveEpilogue::Use_TMA_O;
    static constexpr bool PackGQA = CollectiveMainloop::PackGQA;
    static constexpr int NumProducerThreads = CollectiveMainloop::NumProducerThreads;
    static constexpr bool SameHeadDim = CollectiveMainloop::SameHeadDim;
    static constexpr bool LargeHeadDimV = CollectiveMainloop::LargeHeadDimV;
    static_assert(CollectiveMainloop::LargeHeadDimV == CollectiveEpilogue::LargeHeadDimV);
    using SeqlenInfo_t = typename CollectiveMainloop::SeqlenInfo_t;

    // Mainloop derived types
    using TileShape_MNK_PV = typename CollectiveMainloop::TileShape_MNK_PV;
    using TiledMmaPV = typename CollectiveMainloop::TiledMmaPV;
    using ArchTag = typename CollectiveMainloop::ArchTag;
    using ClusterShape = typename CollectiveMainloop::ClusterShape;
    using MainloopArguments = typename CollectiveMainloop::Arguments;
    using MainloopParams = typename CollectiveMainloop::Params;
    using BarrierQ = std::conditional_t<Use_TMA_Q, cutlass::arch::ClusterTransactionBarrier, cutlass::arch::ClusterBarrier>;

    // Epilogue derived types
    using EpilogueArguments = typename CollectiveEpilogue::Arguments;
    using EpilogueParams = typename CollectiveEpilogue::Params;

    static_assert(ArchTag::kMinComputeCapability >= 90);

    using TileScheduler = TileScheduler_;
    using TileSchedulerArguments = typename flash::TileSchedulerArguments;
    using TileSchedulerParams = typename TileScheduler::Params;

    static constexpr uint32_t NumLoadWarpGroups = 1;
    static constexpr uint32_t NumMmaWarpGroups = CUTE_STATIC_V(size(TiledMmaPV{})) / cutlass::NumThreadsPerWarpGroup;
    static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMmaPV{})) + (NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup);
    static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
    static_assert(NumMmaWarpGroups == 1 || NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);

    /// Register requirement for Load and Math WGs
    // If we use cp.async to load K and V, we need more registers for the producer WG.
    static constexpr uint32_t LoadRegisterRequirement = NumMmaWarpGroups == 1 ? 56 : (NumMmaWarpGroups == 2 ? (Use_TMA_KV ? 24 : 40) : 32);
    static constexpr uint32_t MmaRegisterRequirement = NumMmaWarpGroups == 1 ? 256 : (NumMmaWarpGroups == 2 ? (Use_TMA_KV ? 240 : 232) : 160);
    // If you want to print from the producer warp, you'd need to increase the number of registers
    // Otherwise you'll get CUDA error.
    // static constexpr uint32_t LoadRegisterRequirement = 40;
    // static constexpr uint32_t MmaRegisterRequirement = NumMmaWarpGroups == 2 ? 232 : 152;

    // Kernel level shared memory storage
    // We overlap the shared memory for the mainloop and epilogue. However, we only want smem_o to overlap with smem_v
    // and nothing else, so we'll pad in case sizeof(smem_o) > sizeof(smem_v).
    static constexpr int mainloop_smem_padding_ = int(sizeof(typename CollectiveEpilogue::TensorStorage)) - int(sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_v)));
    static constexpr int mainloop_smem_padding = mainloop_smem_padding_ < 0 ? 0 : mainloop_smem_padding_;
    struct SharedStorage {
        struct TensorStorage : cute::aligned_struct<128, _1> {
            union {
                struct {
                    cute::array<uint32_t, mainloop_smem_padding / sizeof(uint32_t)> padding_;
                    typename CollectiveMainloop::TensorStorage mainloop;
                };
                // We want smem_o to line up with the start of smem_v
                typename CollectiveEpilogue::TensorStorage epilogue;
            };
        } tensors;
        struct PipelineStorage : cute::aligned_struct<16, _1> {
            alignas(16) BarrierQ barrier_Q;
            alignas(16) BarrierQ barrier_Qv;
            alignas(16) cutlass::arch::ClusterBarrier barrier_O;
            alignas(16) typename CollectiveMainloop::MainloopPipelineK::SharedStorage pipeline_k;
            alignas(16) typename CollectiveMainloop::MainloopPipelineV::SharedStorage pipeline_v;
            alignas(16) typename CollectiveMainloop::MainloopPipelineVt::SharedStorage pipeline_vt;
            alignas(16) typename CollectiveMainloop::MainloopPipelineKVNew::SharedStorage pipeline_k_new;
            alignas(16) typename CollectiveMainloop::MainloopPipelineKVNew::SharedStorage pipeline_v_new;
            alignas(16) typename TileScheduler::SharedStorage smem_scheduler;
        } pipelines;

    };

    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    // Device side arguments
    struct Arguments {
        MainloopArguments mainloop{};
        EpilogueArguments epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerArguments scheduler{};
    };

    // Kernel entry point API
    struct Params {
        MainloopParams mainloop{};
        EpilogueParams epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerParams scheduler{};
    };

    //
    // Methods
    //

    // Convert to underlying arguments. In this case, a simple copy for the aliased type.
    static
    Params
    to_underlying_arguments(Arguments const& args) {
        CUTLASS_TRACE_HOST("to_underlying_arguments():");

        // Get SM count if needed, otherwise use user supplied SM count
        int sm_count = args.hw_info.sm_count;
        if (sm_count <= 0) {
            CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
                "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
            sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
        }

        CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

        cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};
        return {
            CollectiveMainloop::to_underlying_arguments(args.mainloop),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue),
            hw_info,
            TileScheduler::to_underlying_arguments(args.scheduler)
        };
    }

    // Computes the kernel launch grid shape based on runtime parameters
    static dim3
    get_grid_shape(Params const& params) {
        return TileScheduler::get_grid_shape(params.scheduler, params.hw_info.sm_count);
    }

    static dim3
    get_block_shape() {
        return dim3(MaxThreadsPerBlock, 1, 1);
    }

    /**
     * FlashAttention 前向内核算子 - SM90 内核执行的主入口点
     * 
     * 该函数使用生产者-消费者模式实现 FlashAttention 前向传播：
     * - 生产者线程组 (WG0)：将 Q、K、V 张量从全局内存加载到共享内存
     * - 消费者线程组 (WG1+)：执行矩阵乘法和注意力计算
     * 
     * 当前配置：page_size=1, Use_TMA_KV=false
     * - 不使用 TMA (Tensor Memory Accelerator) 加载 K/V，使用异步拷贝
     * - 页面大小为 1，意味着每个页面只包含一个数据块
     * 
     * @param params 内核参数，包含主循环、尾声、硬件信息和调度器参数
     * @param smem_buf 用于张量存储和流水线同步的共享内存缓冲区
     */
    CUTLASS_DEVICE
    void
    operator()(Params const& params, char* smem_buf) {

        // 线程组织和内存布局的常量定义
        static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
        // 1 * 128
        static constexpr int MmaThreadOffset = NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
        static constexpr int kBlockM = get<0>(TileShape_MNK_PV{});

        // 不同数据流的流水线类型定义
        using MainloopPipelineK = typename CollectiveMainloop::MainloopPipelineK;
        using MainloopPipelineV = typename CollectiveMainloop::MainloopPipelineV;
        using MainloopPipelineVt = typename CollectiveMainloop::MainloopPipelineVt;  // V 转置时使用
        using MainloopPipelineKVNew = typename CollectiveMainloop::MainloopPipelineKVNew;  // AppendKV 特性使用
        using PipelineState = typename CollectiveMainloop::PipelineState;
        using PipelineParamsK = typename MainloopPipelineK::Params;
        using PipelineParamsV = typename MainloopPipelineV::Params;
        using PipelineParamsVt = typename MainloopPipelineVt::Params;
        using PipelineParamsKVNew = typename MainloopPipelineKVNew::Params;

        // 获取共享内存存储引用
        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        // 线程协调变量
        int const lane_predicate = cute::elect_one_sync();
        int const warp_idx = cutlass::canonical_warp_idx_sync();

        // TMA 描述符预取 - 每个块由单个线程执行一次
        // 注意：当前配置 Use_TMA_KV=false，所以 K/V 不使用 TMA 预取
        if (warp_idx == 0 && lane_predicate) {
            CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
            CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
        }

        // 确定线程在线程组组织中的角色
        int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
        int warp_group_idx = cutlass::canonical_warp_group_idx();

        // 初始化同步屏障 - 每个块执行一次
        if (warp_idx == 0 && lane_predicate) {
            // Q 张量同步屏障
            shared_storage.pipelines.barrier_Q.init(Use_TMA_Q ? 1 : NumProducerThreads /*numThreads*/);
            // Qv 张量同步屏障（如果启用 HasQv 特性）
            if constexpr (HasQv) {
                shared_storage.pipelines.barrier_Qv.init(Use_TMA_Q ? 1 : NumProducerThreads /*numThreads*/);
            }
            // 输出张量同步屏障
            shared_storage.pipelines.barrier_O.init(size(ClusterShape{}) * (Use_TMA_O ? 1 : NumMmaThreads) /*numThreads*/);
        }

        // 配置 K 张量加载的流水线参数
        // pipeline_k 会调用 cutlass::arch::fence_barrier_init()
        PipelineParamsK pipeline_params_k;
        pipeline_params_k.role = warp_group_idx == 0
            ? MainloopPipelineK::ThreadCategory::Producer
            : MainloopPipelineK::ThreadCategory::Consumer;
        if constexpr (Use_TMA_KV) {
            // TMA 加载配置（当前不使用）
            pipeline_params_k.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
            pipeline_params_k.is_leader = warp_group_thread_idx == 0;
            pipeline_params_k.num_consumers = !LargeHeadDimV ? NumMmaThreads : cutlass::NumThreadsPerWarpGroup;
        } else {
            // 异步拷贝加载配置（当前使用此模式）
            pipeline_params_k.consumer_arv_count = !LargeHeadDimV ? NumMmaThreads : cutlass::NumThreadsPerWarpGroup;
            pipeline_params_k.producer_arv_count = NumProducerThreads;
        }

        // 配置 V 张量的流水线参数（以 K 参数为基础）
        static_assert(is_same_v<PipelineParamsK, PipelineParamsVt>);
        PipelineParamsVt pipeline_params_vt = pipeline_params_k;
        if constexpr (Use_TMA_KV && !SameHeadDim) {
            pipeline_params_vt.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;
            if constexpr (LargeHeadDimV) { pipeline_params_vt.num_consumers = NumMmaThreads; }
        } else {
            if constexpr (LargeHeadDimV) { pipeline_params_vt.consumer_arv_count = NumMmaThreads; }
        }

        // 初始化 K 张量流水线
        // 由于 Use_TMA_KV=false，使用异步拷贝模式
        MainloopPipelineK pipeline_k = [&] {
            if constexpr (Use_TMA_KV) {
                return MainloopPipelineK(shared_storage.pipelines.pipeline_k, pipeline_params_k, ClusterShape{});
            } else {
                return MainloopPipelineK(shared_storage.pipelines.pipeline_k, pipeline_params_k);
            }
        }();

        // 初始化 V 张量流水线
        MainloopPipelineV pipeline_v = [&] {
            if constexpr (!Transpose_V) {
                // 直接加载 V，不进行转置
                static_assert(is_same_v<PipelineParamsK, PipelineParamsV>);
                if constexpr (Use_TMA_KV) {
                    return MainloopPipelineV(shared_storage.pipelines.pipeline_v, pipeline_params_vt, ClusterShape{});
                } else {
                    // 当前配置：使用异步拷贝加载 V
                    return MainloopPipelineV(shared_storage.pipelines.pipeline_v, pipeline_params_vt);
                }
            } else {
                // 带转置的 V 加载（例如，FP8 行主序 V）
                PipelineParamsV pipeline_params_v;
                pipeline_params_v.role = warp_group_idx == 0
                    ? MainloopPipelineV::ThreadCategory::Producer
                    : MainloopPipelineV::ThreadCategory::Consumer;
                pipeline_params_v.producer_arv_count = NumProducerThreads;
                pipeline_params_v.consumer_arv_count = NumMmaThreads;
                return MainloopPipelineV(shared_storage.pipelines.pipeline_v, pipeline_params_v);
            }
        }();

        // 初始化 V 转置流水线
        // 如果需要转置 V（例如 FP8 且 V 是行主序），使用 pipeline_vt 进行 TMA，
        // 然后生产者线程组从 pipeline_vt 读取并写入 pipeline_v。
        // 如果不需要转置 V，使用 pipeline_v 进行 TMA，pipeline_vt 不会被使用。
        // 从技术上讲，对于 pipeline_params_vt，WG0 的 warp0 是生产者，WG0 的所有线程是消费者。
        // 但是，线程角色在流水线实现中并未使用。
        MainloopPipelineVt pipeline_vt = [&] {
            if constexpr (Use_TMA_KV) {
                pipeline_params_vt.num_consumers = NumProducerThreads; // TMA_V 仅被生产者线程组消费
                return MainloopPipelineVt(shared_storage.pipelines.pipeline_vt, pipeline_params_vt, ClusterShape{});
            } else {
                // 当前配置：异步拷贝模式
                pipeline_params_vt.consumer_arv_count = NumProducerThreads; // TMA_V 仅被生产者线程组消费
                return MainloopPipelineVt(shared_storage.pipelines.pipeline_vt, pipeline_params_vt);
            }
        }();

        // 配置 AppendKV 特性的流水线（新的 K/V 张量）
        PipelineParamsKVNew pipeline_params_kv_new;
        pipeline_params_kv_new.role = warp_group_idx == 0
            ? MainloopPipelineKVNew::ThreadCategory::Producer
            : MainloopPipelineKVNew::ThreadCategory::Consumer;
        pipeline_params_kv_new.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
        pipeline_params_kv_new.is_leader = warp_group_thread_idx == 0;
        pipeline_params_kv_new.num_consumers = NumMmaThreads;
        
        // 初始化新 K 流水线（条件性依赖于 AppendKV 特性）
        auto pipeline_k_new = cute::conditional_return<AppendKV>(MainloopPipelineKVNew(shared_storage.pipelines.pipeline_k_new, pipeline_params_kv_new, ClusterShape{}), nullptr);
        if constexpr (!SameHeadDim) {
            pipeline_params_kv_new.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;
        }
        // 初始化新 V 流水线（条件性依赖于 AppendKV 特性）
        auto pipeline_v_new = cute::conditional_return<AppendKV>(MainloopPipelineKVNew(shared_storage.pipelines.pipeline_v_new, pipeline_params_kv_new, ClusterShape{}), nullptr);

        // 初始化主循环和尾声集合操作
        CollectiveMainloop mainloop;
        CollectiveEpilogue epilogue;

        // 确保流水线初始化在集群中的所有线程间可见
        // 需要保证流水线初始化对集群中所有生产者和消费者块可见
        if constexpr (size(ClusterShape{}) > 1) {
            cute::cluster_arrive_relaxed();
            cute::cluster_wait();
        } else {
            __syncthreads();
        }

        // 初始化用于工作分配的瓦片调度器
        TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.pipelines.smem_scheduler));

        // =====================================================================
        // Flash Attention 双 Warpgroup 协作架构
        // =====================================================================
        // Flash Attention 使用生产者-消费者模式的双 warpgroup 协作：
        //   - Warpgroup 0 (生产者): 负责从全局内存异步加载 Q、K、V 张量到共享内存
        //   - Warpgroup 1 (消费者): 负责从共享内存读取数据并执行矩阵乘法和注意力计算
        // 
        // 关键同步点：
        //   1. QueryEmpty: 确保共享内存中的 Q 区域可以安全写入
        //   2. AppendKV: 协调 KV_new 张量的加载和存储（用于增量注意力）
        //   3. barrier_O: 确保输出张量写入完成，V 区域可以重用
        //   4. TransposeBarrier: 协调 V 张量的转置操作
        // =====================================================================

        if (warp_group_idx == 0) {  // 生产者线程组 - 处理数据加载
            // ===================================================================
            // 生产者 Warpgroup (索引 0) - 数据加载管道
            // ===================================================================
            // 职责：
            // 1. 异步加载 Q、K、V 张量从全局内存到共享内存
            // 2. 管理多级流水线状态，实现重叠计算和内存访问
            // 3. 处理页表管理（用于 PagedAttention）
            // 4. 协调 V 张量转置操作（如果需要）
            // ===================================================================
            
            // 释放加载操作不需要的寄存器，为生产者流水线腾出空间
            cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

            // AppendKV 和主注意力的流水线不同，因为例如主注意力可能使用 cp.async 
            // 加载 KV（如果是 PagedKVNonTMA），而 AppendKV 总是使用 TMA 加载 KV_new。
            // 由于流水线状态不同，我们必须手动同步以确保两个流水线在访问 smem_k 和 smem_v 时不会竞争。
            
            // 初始化两个独立的流水线状态：
            // - smem_pipe_write: 主注意力流水线（Q@K^T 和 Attention@V）
            // - smem_pipe_write_new: AppendKV 流水线（处理新增的 K/V 张量）
            PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipelineK>();
            PipelineState smem_pipe_write_new = cutlass::make_producer_start_state<MainloopPipelineKVNew>();
            int work_idx = 0;
            int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
            static constexpr bool SingleProducerWarp = NumProducerThreads == cutlass::NumThreadsPerWarp;
            
            // 处理单个 vs 多个生产者 warp 配置
            // 在多 warp 生产者模式下，只有第一个 warp 执行关键的加载逻辑
            if constexpr (SingleProducerWarp) {
                if (warp_idx_in_warpgroup != 0) { return; }
            }
            if (!SingleProducerWarp && warp_idx_in_warpgroup != 0) { scheduler.init_consumer(); }

            // 等待依赖网格完成（用于多内核场景，如分片注意力）
            cutlass::arch::wait_on_dependent_grids();

            // ===================================================================
            // 主生产者循环：为每个工作瓦片加载 Q、K、V 张量
            // ===================================================================
            // 工作瓦片调度策略：
            // - page_size=1 意味着每次处理一个页面的数据
            // - 使用动态调度来平衡负载
            // ===================================================================
            for (auto work_tile_info = SingleProducerWarp || warp_idx_in_warpgroup == 0 ? scheduler.template get_initial_work</*IsProducerWarp=*/true>(params.scheduler) : scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler);
                 work_tile_info.is_valid(params.scheduler);
                 work_tile_info = SingleProducerWarp || warp_idx_in_warpgroup == 0 ? scheduler.template get_next_work</*IsProducerWarp=*/true>(params.scheduler, work_tile_info) : scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info)) {

                // 提取块坐标并设置序列长度信息
                auto block_coord = work_tile_info.get_block_coord(params.scheduler);
                SeqlenInfo_t seqlen_info{
                    get<2>(block_coord) /*bidb*/,
                    get<0>(params.mainloop.shape_Q),
                    !params.mainloop.ptr_pagetable ? size<0>(params.mainloop.shape_K) : size<0>(params.mainloop.shape_K) * size<1>(params.mainloop.shape_pagetable),
                    get<0>(params.mainloop.shape_K_new),
                    params.mainloop.cu_seqlens_q, params.mainloop.cu_seqlens_k, params.mainloop.cu_seqlens_k_new,
                    params.mainloop.seqused_q, params.mainloop.seqused_k, params.mainloop.leftpad_k,
                    params.mainloop.seqlens_rotary
                };
                
                // ===============================================================
                // AppendKV 特性处理：增量注意力中的新 K/V 张量加载
                // ===============================================================
                // AppendKV 用于增量生成场景，新的 token 对应的 K/V 需要追加到现有序列
                // 这需要特殊的同步机制确保数据一致性
                // ===============================================================
                if constexpr (AppendKV) {
                    bool tile_new_valid = mainloop.load_kv_new(
                        params.mainloop, pipeline_k_new, pipeline_v_new,
                        smem_pipe_write_new, shared_storage, seqlen_info, block_coord, work_idx);
                    if (tile_new_valid) {
                        // ========================================================
                        // 【关键同步点 1】AppendKV Barrier
                        // ========================================================
                        // 目的：确保生产者加载的 KV_new 与消费者存储的 KV_new 正确同步
                        // 参与者：NumMmaThreads (消费者) + NumProducerThreads (生产者)
                        // 机制：Named Barrier - 双向同步点
                        // 
                        // 时序关系：
                        // 1. 生产者加载 KV_new 到共享内存
                        // 2. 在此处同步，确保消费者看到新数据
                        // 3. 消费者将 KV_new 存储到全局内存
                        // 4. 消费者也在相同 barrier 同步
                        // ========================================================
                        // if (threadIdx.x == 0) { printf("Producer: Before sync\n"); }
                        cutlass::arch::NamedBarrier::sync(NumMmaThreads + NumProducerThreads, static_cast<uint32_t>(FwdNamedBarriers::AppendKV) /*id*/);
                        // if (threadIdx.x == 0) { printf("Producer: After sync\n"); }
                    }
                }
                
                // 为下一个工作瓦片设置预取回调，优化内存访问模式
                auto scheduler_prefetch = [&scheduler, &params, &work_tile_info]() {
                    scheduler.prefetch_next_work(params.scheduler, work_tile_info);
                };
                
                // ===============================================================
                // 主要张量加载：Q、K、V 的异步加载流水线
                // ===============================================================
                // 加载策略：
                // - Q: 可以使用 TMA 或 cp.async
                // - K/V: 支持 TMA 和非 TMA 模式
                // - V 转置：根据需要在加载后进行转置
                // 
                // 流水线阶段：
                // 1. TMA/cp.async 发起内存传输
                // 2. 等待传输完成
                // 3. 可选的数据转换（如 V 转置）
                // ===============================================================
                mainloop.load(params.mainloop, pipeline_k, pipeline_v, pipeline_vt, smem_pipe_write,
                                         shared_storage, scheduler_prefetch, seqlen_info, block_coord, work_idx);
            }
            // 完成所有加载操作，清理流水线状态
            mainloop.load_tail(pipeline_k, pipeline_v, pipeline_vt, smem_pipe_write, shared_storage, work_idx);
            
        } else {  // 消费者线程组 - 处理矩阵乘法和注意力计算
            // ===================================================================
            // 消费者 Warpgroup (索引 1) - 计算管道
            // ===================================================================
            // 职责：
            // 1. 从共享内存读取 Q、K、V 张量
            // 2. 执行 Q@K^T 矩阵乘法计算注意力分数
            // 3. 应用 Softmax 归一化
            // 4. 执行 Attention@V 矩阵乘法得到最终输出
            // 5. 处理 FP8 量化、因果掩码、滑动窗口等特性
            // ===================================================================
            
            // 分配 MMA 操作所需的寄存器，为张量核心计算做准备
            cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

            // 初始化张量核心矩阵乘法对象（针对 Attention@V 阶段）
            TiledMmaPV tiled_mma_pv;

            // 从共享内存读取的流水线状态
            // 注意：读取和释放使用相同的流水线状态（与 CUTLASS GEMM 不同）
            PipelineState smem_pipe_read;
            PipelineState smem_pipe_read_new;

            // 初始化消费者调度器和 MMA 计算单元
            scheduler.init_consumer();
            mainloop.mma_init();

            int work_idx = 0;
            // ===================================================================
            // 主消费者循环：处理每个工作瓦片的注意力计算
            // ===================================================================
            CUTLASS_PRAGMA_NO_UNROLL
            for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler);
                 work_tile_info.is_valid(params.scheduler);
                 // get_next_work 将在尾声之前调用，优化流水线
                 ) {
                // 提取块坐标并设置序列信息
                auto block_coord = work_tile_info.get_block_coord(params.scheduler);
                int const bidb = get<2>(block_coord);
                SeqlenInfo_t seqlen_info{
                    bidb,
                    get<0>(params.mainloop.shape_Q),
                    !params.mainloop.ptr_pagetable ? size<0>(params.mainloop.shape_K) : size<0>(params.mainloop.shape_K) * size<1>(params.mainloop.shape_pagetable),
                    get<0>(params.mainloop.shape_K_new),
                    params.mainloop.cu_seqlens_q, params.mainloop.cu_seqlens_k, params.mainloop.cu_seqlens_k_new,
                    params.mainloop.seqused_q, params.mainloop.seqused_k, params.mainloop.leftpad_k,
                    params.mainloop.seqlens_rotary
                };
                
                // ===============================================================
                // AppendKV 特性处理：消费者端的 KV_new 存储
                // ===============================================================
                if constexpr (AppendKV) {
                    bool tile_new_valid = mainloop.store_kv_new(
                        params.mainloop, pipeline_k_new, pipeline_v_new, smem_pipe_read_new,
                        threadIdx.x - MmaThreadOffset, shared_storage, seqlen_info, block_coord);
                    if (tile_new_valid) {
                        // ========================================================
                        // 【关键同步点 2】AppendKV Barrier (消费者端)
                        // ========================================================
                        // 确保全局内存写入在同步前对其他 SM 可见
                        // fence.proxy.async.global 确保写入操作完成并对其他处理器可见
                        // ========================================================
                        // if (threadIdx.x == 128) { printf("Consumer: Before sync\n"); }
                        // 需要此同步以确保消费者的全局内存写入对可能在之后进行 TMA 读取的生产者可见
                        asm volatile ("fence.proxy.async.global;");
                        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + NumProducerThreads, static_cast<uint32_t>(FwdNamedBarriers::AppendKV) /*id*/);
                        // arrive 就足够了，我们不需要 sync。生产者会 sync，这意味着
                        // 在那个 sync 之后我们保证 AppendKV 流水线已完成加载和消费者 smem_k 和 smem_v。
                        // if (threadIdx.x == 128) { printf("Consumer: After sync\n"); }
                    }
                }
                
                // ===============================================================
                // Softmax 缩放因子设置
                // ===============================================================
                // 处理 FP8 量化的反缩放：scale = softmax_scale * q_descale * k_descale
                // 如果有 tanh softcap，缩放将在 tanh 之前完成
                // ===============================================================
                float softmax_scale_log2 = params.mainloop.softmax_scale_log2;
                if constexpr (Is_FP8 && !Has_softcap) {
                    int const bidh = get<1>(block_coord);
                    int const bidh_kv = !PackGQA ? params.mainloop.qhead_per_khead_divmod.divide(bidh) : bidh;
                    float const q_descale = params.mainloop.ptr_q_descale == nullptr ? 1.0f : params.mainloop.ptr_q_descale[bidb * get<0>(params.mainloop.stride_q_descale) + bidh_kv * get<1>(params.mainloop.stride_q_descale)];
                    float const k_descale = params.mainloop.ptr_k_descale == nullptr ? 1.0f : params.mainloop.ptr_k_descale[bidb * get<0>(params.mainloop.stride_k_descale) + bidh_kv * get<1>(params.mainloop.stride_k_descale)];
                    softmax_scale_log2 *= q_descale * k_descale;
                }
                
                // 初始化 Online Softmax 计算单元
                // 支持 Flash Attention 的分块 Softmax 算法
                // <2, 0>
                flash::Softmax<!LargeHeadDimV ? 2 * (2 * kBlockM / NumMmaThreads) : 2, /*Max_offset=*/!Is_FP8 ? 0 : 8> softmax(softmax_scale_log2);
                
                // 初始化注意力输出累加器 (GEMM-II: Attention@V)
                // 
                Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShape_MNK_PV{}));
                bool tile_valid;
                
                // ===============================================================
                // 注意力计算：根据头维度大小选择不同的执行路径
                // ===============================================================
                // 小头维度：标准单 warpgroup MMA
                // 大头维度：跨 warpgroup 分割 MMA (需要额外的 warpgroup)
                // ===============================================================
                if constexpr (!LargeHeadDimV) {
                    // 较小头维度的标准 MMA 路径
                    // 执行完整的注意力计算：Q@K^T -> Softmax -> Attention@V
                    tile_valid = mainloop.mma(
                        params.mainloop, pipeline_k, pipeline_v, smem_pipe_read,
                        tOrO, softmax, threadIdx.x - MmaThreadOffset, work_idx, seqlen_info, block_coord, shared_storage);
                } else {  // 如果 !LargeHeadDimV，mma_pv 可能无法编译
                    // 较大头维度在线程组间分割 MMA
                    // warpgroup 1: 执行 Q@K^T 和 Softmax
                    // warpgroup 2: 执行 Attention@V
                    if (warp_group_idx == 1) {
                        tile_valid = mainloop.mma(
                            params.mainloop, pipeline_k, pipeline_v, smem_pipe_read,
                            tOrO, softmax, threadIdx.x - MmaThreadOffset, work_idx, seqlen_info, block_coord, shared_storage);
                    } else {
                        tile_valid = mainloop.mma_pv(
                            params.mainloop, pipeline_v, smem_pipe_read,
                            tOrO, softmax, threadIdx.x - MmaThreadOffset, seqlen_info, block_coord, shared_storage);
                    }
                }
                
                // 在尾声前准备下一个工作瓦片以优化流水线
                // 在此处执行以便下一个瓦片准备就绪，减少调度开销
                work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info);
                
                // 如果这是最后一个瓦片，为分割注意力启动依赖网格
                if constexpr (Split && Varlen) {
                    if (!work_tile_info.is_valid(params.scheduler)) {  // 最后一个瓦片
                        cutlass::arch::launch_dependent_grids();
                    }
                }
                
                // ===============================================================
                // 结果存储：根据瓦片有效性存储结果或零值
                // ===============================================================
                if (tile_valid) {
                    // 存储计算的注意力输出到全局内存
                    // 包括注意力权重和 LSE (Log-Sum-Exp) 用于后续的 Softmax 合并
                    // if (threadIdx.x == 128) { printf("Before epilogue, bid.x = %d, bid.y = %d, bid.z = %d, m_block = %d, bidb = %d, split_idx = %d\n", blockIdx.x, blockIdx.y, blockIdx.z, m_block, bidb, split_idx); }
                    epilogue.store(params.epilogue, tOrO, softmax.row_sum, shared_storage, tiled_mma_pv,
                                   threadIdx.x - MmaThreadOffset, block_coord);
                } else {
                    // 为无效瓦片写入 0 到 gO 和 -inf 到 gLSE
                    // 这确保了在变长序列、因果掩码等场景下的正确性
                    epilogue.store_zero(params.epilogue, threadIdx.x - MmaThreadOffset, block_coord);
                }
            }
            // 完成尾声操作，确保所有输出写入完成
            epilogue.store_tail();
        }

    }

};

} // namespace flash

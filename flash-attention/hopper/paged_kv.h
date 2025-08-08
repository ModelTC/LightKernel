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
 * 分页KV缓存管理器模板类
 * 
 * @tparam kBlockN 块大小（N维度）
 * @tparam kHeadDim K张量的头维度
 * @tparam kHeadDimV V张量的头维度  
 * @tparam NumThreads 线程数量
 * @tparam Element 数据元素类型
 * @tparam KV_Same_Iter 是否在同一迭代中处理KV，默认false
 * @tparam LoadsPerRow_LB 每行加载次数的下界，用于旋转位置编码优化
 * 
 * 功能说明：
 * - 如果 KV_Same_Iter=false，执行顺序为：load_page_table(0), load_K(0), load_page_table(1), 
 *   load_K(1), load_V(0), load_page_table(2), load_K(2), load_V(1), 等等
 * - 需要为前一次迭代计算V指针
 * - LoadsPerRow_LB 是K方向每行加载次数的下界，对于旋转位置编码很有用，
 *   我们希望每个线程至少有2次每行加载
 */
// <64, 64, 64, 128, float16, true>
template <int kBlockN, int kHeadDim, int kHeadDimV, int NumThreads, typename Element, bool KV_Same_Iter=false, int LoadsPerRow_LB=1>
struct PagedKVManager {
    
    // 静态常量定义
    static constexpr bool SameHeadDim = (kHeadDim == kHeadDimV);  // K和V是否有相同的头维度 false
    static constexpr int kHeadDimGCD = cute::gcd(kHeadDim, kHeadDimV);  // K和V头维度的最大公约数 64

    // 对于分页KV，我们使用CpAsync而不是TMA，因为TMA在这里不工作
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);  // 每次加载的全局内存元素数 128 / 16 = 8
    static_assert(kHeadDimGCD % kGmemElemsPerLoad == 0, "Headdim and HeaddimV must be a multiple of kGmemElemsPerLoad");
    
    // 我们希望每"行"有64个元素（128字节，即1个缓存行）
    // 例如，如果hdim=128，我们希望每个线程在M方向有4次加载，在K方向有2次向量化加载
    // 在PackGQA的情况下，这减少了我们需要调用divmod的次数
    static_assert(kHeadDimGCD % LoadsPerRow_LB == 0, "Headdim and HeaddimV must be a multiple of LoadsPerRow_LB");
    static constexpr int kBytePerRow = kHeadDimGCD / LoadsPerRow_LB * sizeof(Element);  // 每行字节数 128
    static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);  // 全局内存块K大小 128
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;  // 每行的全局内存线程数 128 / 8 = 16
    static_assert(NumThreads % kGmemThreadsPerRow == 0, "NumThreads must be a multiple of kGmemThreadsPerRow");
    
    // 我们假设加载同一行的线程在同一个warp中，这是对分页KV的优化
    // 这些线程共享相同的页表条目并共享计算分页K和分页V指针的工作
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0, "kGmemThreadsPerRow must divide NumThreadsPerWarp");
    
    // 定义各种类型别名，用于内存操作和张量布局
    using GmemCopyAtomCpAsync = cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, Element>;
    using GmemLayoutAtomKVCpAsync = Layout<Shape <Int<NumThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                           Stride<Int<kGmemThreadsPerRow>, _1>>; // (128 / 16, 16) (8, 16)
    using GmemTiledCopyKVCpAsync = decltype(
        make_tiled_copy(GmemCopyAtomCpAsync{},
                        GmemLayoutAtomKVCpAsync{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // 值布局，每次加载8或16个值
    using GmemTiledCopyKVStore = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtomKVCpAsync{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // 值布局，每次加载8或16个值

    // 张量形状和步长定义
    using ShapeKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // KV张量形状：(seqlen, d, head, batch)
    using StrideKV = cute::Stride<int64_t, _1, int64_t, int64_t>;     // KV张量步长
    using ShapePageTable = cute::Shape<int32_t, int32_t>;  // 页表形状：(batch, max_num_pages_per_seq)
    using StridePageTable = cute::Stride<int64_t, _1>;     // 页表步长

    // 张量类型定义
    using TensorPageTable = decltype(make_tensor(make_gmem_ptr(static_cast<int const*>(nullptr)), ShapePageTable{}, StridePageTable{})(int(0), _));
    using TensorKV = decltype(make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapeKV{}, StrideKV{})(_, _, int(0), _));
    using GmemThrCopyKVCpAsync = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)));
    using TensortKcK = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{})));
    using TensortKpK = decltype(make_tensor<bool>(make_shape(size<1>(TensortKcK{}), size<2>(TensortKcK{})), Stride<_0, _1>{}));
    using TensortVcV = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{})));
    using TensortVpV = decltype(make_tensor<bool>(make_shape(size<1>(TensortVcV{}), size<2>(TensortVcV{})), Stride<_0, _1>{}));

    // 对于分页KV，计算每个页表条目的K和V指针是昂贵的，因为需要int64_t算术
    // 我们通过让线程分担这项工作来优化
    // 通常每行有8个线程加载（例如hdim 64和128），每个线程需要为hdim 128和kBlockN = 176的情况加载11行
    // 所以这8个线程中的每一个将为11/8=2行计算K_ptr和V_ptr
    // 然后我们使用__shfl_sync将指针广播给warp中的其他线程
    static_assert(CUTE_STATIC_V(size<1>(TensortKcK{})) == CUTE_STATIC_V(size<1>(TensortVcV{})));
    static constexpr int kPageEntryPerThread = cute::ceil_div(size<1>(TensortKcK{}), kGmemThreadsPerRow);  // 每个线程的页条目数
    using TensorPageOffset = decltype(make_tensor<cute::tuple<int, int>>(Shape<Int<kPageEntryPerThread>>{}));  // 页偏移张量
    using TensorKVPtr = decltype(make_tensor<Element*>(Shape<Int<kPageEntryPerThread>>{}));  // KV指针张量

    // 成员变量
    GmemTiledCopyKVCpAsync gmem_tiled_copy_kv;              // 全局内存分块复制KV
    cutlass::FastDivmod const &page_size_divmod;           // 页大小快速除法模运算
    cutlass::FastDivmod const &blockN_per_page_size_divmod; // 每页块N大小的快速除法模运算
    int const thread_idx;                                   // 线程索引
    int const seqlen_k;                                     // K序列长度
    int const leftpad_k;                                    // K的左填充
    int const* const ptr_page_table;                       // 页表指针
    GmemThrCopyKVCpAsync const gmem_thr_copy_kv;           // 全局内存线程复制KV
    TensorPageTable mPageTable;                            // 页表张量
    TensorKV mK_paged, mV_paged;                           // 分页K和V张量
    TensortKpK tKpK;                                       // K谓词张量
    TensortVpV tVpV;                                       // V谓词张量
    TensorPageOffset tPrPageOffset;                        // 页偏移张量
    TensorKVPtr tPrVPtr;                                   // V指针张量
    int bidb_kv_idx, bidb_kv_idx_prev, n_block_idx, n_block_idx_prev;  // 批次和块索引，仅用于TMA

    /**
     * 构造函数：初始化分页KV管理器
     * 
     * @param ptr_page_table_ 页表指针
     * @param shape_pagetable 页表形状
     * @param stride_pagetable 页表步长
     * @param ptr_K K张量指针
     * @param shape_K K张量形状
     * @param stride_K K张量步长
     * @param ptr_V V张量指针
     * @param headdim_v V的头维度
     * @param stride_V V张量步长
     * @param page_size_divmod 页大小除法模运算器
     * @param blockN_per_page_size_divmod 每页块N大小除法模运算器
     * @param bidb 批次索引
     * @param bidh 头索引
     * @param thread_idx 线程索引
     * @param seqlen_k K序列长度
     * @param leftpad_k K的左填充
     * @param bidb_kv_idx KV批次索引
     */
    CUTLASS_DEVICE
    PagedKVManager(int const* const ptr_page_table_,
                   ShapePageTable const &shape_pagetable, StridePageTable const &stride_pagetable,
                   Element* const ptr_K, ShapeKV const &shape_K, StrideKV const &stride_K,
                   Element* const ptr_V, int const headdim_v, StrideKV const &stride_V,
                   cutlass::FastDivmod const &page_size_divmod,
                   cutlass::FastDivmod const &blockN_per_page_size_divmod,
                   int const bidb, int const bidh, int const thread_idx, int const seqlen_k, int const leftpad_k,
                   int bidb_kv_idx
                   )
        : page_size_divmod(page_size_divmod)
        , blockN_per_page_size_divmod(blockN_per_page_size_divmod)
        , thread_idx(thread_idx)
        , seqlen_k(seqlen_k)
        , leftpad_k(leftpad_k)
        , ptr_page_table(ptr_page_table_)
        , gmem_thr_copy_kv(gmem_tiled_copy_kv.get_thread_slice(thread_idx))
        , bidb_kv_idx(bidb_kv_idx)
        , bidb_kv_idx_prev(bidb_kv_idx)

    {
        // 初始化页表张量
        mPageTable = make_tensor(make_gmem_ptr(ptr_page_table), shape_pagetable, stride_pagetable)(bidb, _);
        // 初始化分页K张量
        mK_paged = make_tensor(make_gmem_ptr(ptr_K), shape_K, stride_K)(_, _, bidh, _);
        // 构建V张量形状并初始化分页V张量
        auto shape_V = make_shape(get<0>(shape_K), headdim_v, get<2>(shape_K), get<3>(shape_K));
        mV_paged = make_tensor(make_gmem_ptr(ptr_V), shape_V, stride_V)(_, _, bidh, _);
        
        // 初始化K谓词张量
        tKpK = make_tensor<bool>(make_shape(size<1>(TensortKcK{}), size<2>(TensortKcK{})), Stride<_0, _1>{});
        Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
        // 设置K谓词：检查是否在有效范围内
        #pragma unroll
        for (int k = 0; k < size<1>(tKpK); ++k) { tKpK(_0{}, k) = get<1>(tKcK(_0{}, _0{}, k)) < get<1>(shape_K); }
        
        // 初始化V谓词张量
        Tensor tVpV_ = make_tensor<bool>(make_shape(size<1>(TensortVcV{}), size<2>(TensortVcV{})), Stride<_0, _1>{});
        Tensor cV = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tVcV = gmem_thr_copy_kv.partition_S(cV);
        // 设置V谓词：检查是否在有效范围内
        #pragma unroll
        for (int k = 0; k < size<1>(tVpV_); ++k) { tVpV_(_0{}, k) = get<1>(tVcV(_0{}, _0{}, k)) < get<1>(shape_V); }
        // 根据头维度是否相同选择谓词
        tVpV = cute::conditional_return<SameHeadDim>(tKpK, tVpV_);
    };

    /**
     * 加载页表（非TMA版本）
     * 
     * @tparam Seqlenk_mask 是否应用序列长度掩码
     * @tparam First_iter 是否为第一次迭代
     * @param n_block 块索引
     * 
     * 说明：非聚合的全局内存加载是故意的，这样每个线程只加载它需要的页表条目，
     * 我们不需要warp之间的同步。假设每行8个线程，176行，那么行0-175由
     * 线程0,8,16,...,120,1,9,...,121,2,10,...,122等加载。
     */
    template <bool Seqlenk_mask=false, bool First_iter=false>
    CUTLASS_DEVICE
    void load_page_table(const int n_block) {
        #pragma unroll
        for (int i = 0; i < kPageEntryPerThread; ++i) {
            // 计算当前线程要处理的行索引
            int const row = i * NumThreads + (thread_idx % kGmemThreadsPerRow) * (NumThreads / kGmemThreadsPerRow) + (thread_idx / kGmemThreadsPerRow);
            int const row_idx = n_block * kBlockN + row;
            int page_idx, page_offset;
            // 使用快速除法模运算计算页索引和页内偏移
            page_idx = page_size_divmod.divmod(page_offset, row_idx + leftpad_k);
            // 添加条件 (i + 1) * NumThreads <= kBlockN，因为这是row的上界且在编译时已知
            // 当例如kBlockN = 176且i = 0时，它避免了分支
            int const page = ((i + 1) * NumThreads <= kBlockN || row < kBlockN) && (!Seqlenk_mask || row_idx < seqlen_k) ? mPageTable[page_idx] : 0;
            tPrPageOffset[i] = {page, page_offset};
            // 调试输出（已注释）
            // if (cute::thread0()) { printf("row = %d, page_idx = %d, page_offset = %d, page = %d, leftpad_k = %d, seqlen_k = %d\n", row, page_idx, page_offset, page, leftpad_k, seqlen_k); }
        }
        // 如果是第一次迭代且KV不在同一迭代中处理，计算V指针
        if constexpr (First_iter && !KV_Same_Iter) { compute_V_ptr(); }
    };

    /**
     * 加载页表（TMA版本）
     * 
     * @tparam First_iter 是否为第一次迭代
     * @param n_block 块索引
     * 
     * 说明：我们要求页大小是kBlockN的倍数，且没有leftpad_k
     */
    template <bool First_iter=false>
    CUTLASS_DEVICE
    void load_page_table_TMA(const int n_block) {
        if (ptr_page_table) {
            // 使用页表计算KV批次索引
            bidb_kv_idx = mPageTable[blockN_per_page_size_divmod.divmod(n_block_idx, n_block)];
        } else {
            // 没有页表时直接使用块索引
            n_block_idx = n_block;
        }
        // 如果是第一次迭代且KV不在同一迭代中处理，保存当前索引
        if constexpr (First_iter && !KV_Same_Iter) {
            bidb_kv_idx_prev = bidb_kv_idx;
            n_block_idx_prev = n_block_idx;
        }
    };

    /**
     * 获取K张量TMA操作的索引
     * @return 返回块索引和批次索引的元组
     */
    CUTLASS_DEVICE
    cute::tuple<int, int> get_indices_for_K_TMA() {
        return {n_block_idx, bidb_kv_idx};
    };

    /**
     * 获取V张量TMA操作的索引
     * @return 返回块索引和批次索引的元组
     * 
     * 说明：如果KV在同一迭代中处理，返回当前索引；否则返回前一次的索引并更新
     */
    CUTLASS_DEVICE
    cute::tuple<int, int> get_indices_for_V_TMA() {
        if constexpr (KV_Same_Iter) {
            return {n_block_idx, bidb_kv_idx};
        } else {
            cute::tuple<int, int> const indices = {n_block_idx_prev, bidb_kv_idx_prev};
            bidb_kv_idx_prev = bidb_kv_idx;
            n_block_idx_prev = n_block_idx;
            return indices;
        }
    };

    /**
     * 计算K张量的指针
     * @return 返回K指针张量
     */
    CUTLASS_DEVICE
    TensorKVPtr compute_K_ptr() {
        Tensor tPrKPtr = make_tensor<Element*>(Shape<Int<kPageEntryPerThread>>{});
        #pragma unroll
        for (int i = 0; i < kPageEntryPerThread; ++i) {
            auto [page, page_offset] = tPrPageOffset[i];
            // 根据页和页内偏移计算K张量指针
            tPrKPtr[i] = &mK_paged(page_offset, _0{}, page);
        }
        return tPrKPtr;
    };

    /**
     * 计算V张量的指针
     */
    CUTLASS_DEVICE
    void compute_V_ptr() {
        #pragma unroll
        for (int i = 0; i < kPageEntryPerThread; ++i) {
            auto [page, page_offset] = tPrPageOffset[i];
            // 根据页和页内偏移计算V张量指针
            tPrVPtr[i] = &mV_paged(page_offset, _0{}, page);
        }
    };

    /**
     * 加载K张量数据
     * 
     * @tparam Seqlenk_mask 是否应用序列长度掩码
     * @tparam TensorK K张量类型
     * @param n_block 块索引
     * @param sK 共享内存中的K张量
     */
    template <bool Seqlenk_mask=false, typename TensorK>
    CUTLASS_DEVICE
    void load_K(const int n_block, TensorK &&sK) {
        // 检查是否需要边界检查以确保行不超过kBlockN
        static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtomKVCpAsync{})) == 0;

        // 计算K指针
        Tensor tPrKPtr = compute_K_ptr();

        // 仅用于索引计算，因为线程0的所有索引在编译时都是已知的
        auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
        Tensor tKsK = gmem_thr_copy_kv.partition_D(sK);
        Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        // 使用恒等布局重复分区
        Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
        Tensor t0KcK = gmem_thr0_copy_kv.partition_S(cK);

        // 我们希望使用thread0的行索引进行比较，因为这在编译时是已知的
        // 所以我们从限制中减去这个线程的第一行索引 (get<0>(tKcK(_0{}, _0{}, _0{})))
        int const seqlenk_row_limit = -int(get<0>(tKcK(_0{}, _0{}, _0{}))) + (EvenN
            ? seqlen_k - n_block * kBlockN
            : (!Seqlenk_mask ? kBlockN : std::min(seqlen_k - n_block * kBlockN, kBlockN)));
        #pragma unroll
        for (int m = 0; m < size<1>(tKsK); ++m) {
            // 确定是否应该加载这一行
            bool const should_load = EvenN
                ? (!Seqlenk_mask || get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit)
                : get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit;
            // 使用warp shuffle获取K指针（同一warp内线程共享指针计算）
            Element const* k_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrKPtr(m / kGmemThreadsPerRow)), (m % kGmemThreadsPerRow), kGmemThreadsPerRow));
            Tensor mK_paged_cur = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<kHeadDim>>{});
            Tensor mK_paged_cur_copy = cute::tiled_divide(mK_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
            if (should_load) {
                #pragma unroll
                for (int k = 0; k < size<2>(tKsK); ++k) {
                    int const ki = get<1>(tKcK(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    // 使用谓词控制的复制操作
                    cute::copy(gmem_tiled_copy_kv.with(tKpK(_0{}, k)), mK_paged_cur_copy(_, ki), tKsK(_, m, k));
                }
            }  // 不需要清理共享内存的其余部分，因为我们会屏蔽分数
        }
    };

    /**
     * 加载V张量数据
     * 
     * @tparam Seqlenk_mask 是否应用序列长度掩码
     * @tparam TensorV V张量类型
     * @param n_block 块索引
     * @param sV 共享内存中的V张量
     */
    template <bool Seqlenk_mask=false, typename TensorV>
    CUTLASS_DEVICE
    void load_V(const int n_block, TensorV &&sV) {
        // 检查是否需要边界检查以确保行不超过kBlockN
        static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtomKVCpAsync{})) == 0;

        // 如果KV在同一迭代中处理，计算V指针
        if constexpr (KV_Same_Iter) { compute_V_ptr(); }
        // 仅用于索引计算，因为线程0的所有索引在编译时都是已知的
        auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
        Tensor tVsV = gmem_thr_copy_kv.partition_D(sV);
        Tensor cV = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        // 使用恒等布局重复分区
        Tensor tVcV = gmem_thr_copy_kv.partition_S(cV);
        Tensor t0VcV = gmem_thr0_copy_kv.partition_S(cV);

        int const seqlenk_row_limit = seqlen_k - n_block * kBlockN - get<0>(tVcV(_0{}, _0{}, _0{}));
        #pragma unroll
        for (int m = 0; m < size<1>(tVsV); ++m) {
            // 依赖cp.async清理越界的共享内存比直接调用cute::clear更快
            // 如果!EvenN，我们必须小心不要写入超过`kBlockN`的共享内存
            // 如果kBlockN不能整除分块复制，只有最后一个`m`需要检查
            if (EvenN || m < size<1>(tVsV) - 1 || get<0>(tVcV(_0{}, m, _0{})) < kBlockN) {
                bool const should_load = !Seqlenk_mask || get<0>(t0VcV(_0{}, m, _0{})) < seqlenk_row_limit;
                // 使用warp shuffle获取V指针
                Element const* v_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrVPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
                Tensor mV_paged_cur = make_tensor(make_gmem_ptr(v_ptr), Shape<Int<kHeadDimV>>{});
                Tensor mV_paged_cur_copy = cute::tiled_divide(mV_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
                #pragma unroll
                for (int k = 0; k < size<2>(tVsV); ++k) {
                    int const ki = get<1>(tVcV(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    // 使用谓词和加载条件控制的复制操作
                    cute::copy(gmem_tiled_copy_kv.with(tVpV(_0{}, k) && should_load), mV_paged_cur_copy(_, ki), tVsV(_, m, k));
                }
            }
        }
        // 如果KV不在同一迭代中处理，为下次迭代计算V指针
        if constexpr (!KV_Same_Iter) { compute_V_ptr(); }
    };

    /**
     * 存储K张量数据到分页内存
     * 
     * @tparam TensorK K张量类型
     * @param n_block 块索引
     * @param tKrK 要存储的K张量数据
     */
    template <typename TensorK>
    CUTLASS_DEVICE
    void store_K(const int n_block, TensorK &&tKrK) {
        Tensor tPrKPtr = compute_K_ptr();
        // 我们使用与GmemTiledCopyKVCpAsync相同的分区（用于加载）
        // 仅用于索引计算，因为线程0的所有索引在编译时都是已知的
        auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
        Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        // 使用恒等布局重复分区
        Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
        Tensor t0KcK = gmem_thr0_copy_kv.partition_S(cK);

        GmemTiledCopyKVStore gmem_tiled_copy_kv_store;
        // 我们希望使用thread0的行索引进行比较，因为这在编译时是已知的
        // 所以我们从限制中减去这个线程的第一行索引 (get<0>(tKcK(_0{}, _0{}, _0{})))
        int const seqlenk_row_limit = std::min(seqlen_k - n_block * kBlockN, kBlockN) - get<0>(tKcK(_0{}, _0{}, _0{}));
        // 调试输出（已注释）
        // if (threadIdx.x == 128) { printf("bidx = %d, bidy = %d, bidz = %d, seqlen_k = %d, seqlenk_row_limit = %d\n", blockIdx.x, blockIdx.y, blockIdx.z, seqlen_k, seqlenk_row_limit); }
        #pragma unroll
        for (int m = 0; m < size<1>(tKrK); ++m) {
            bool const should_load = get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit;
            // 使用warp shuffle获取K指针
            Element* k_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrKPtr(m / kGmemThreadsPerRow)), (m % kGmemThreadsPerRow), kGmemThreadsPerRow));
            Tensor mK_paged_cur = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<kHeadDim>>{});
            Tensor mK_paged_cur_copy = cute::tiled_divide(mK_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
            if (should_load) {
                #pragma unroll
                for (int k = 0; k < size<2>(tKrK); ++k) {
                    int const ki = get<1>(tKcK(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    // 只有在谓词为真时才存储
                    if (tKpK(_0{}, k)) {
                        cute::copy(gmem_tiled_copy_kv_store, tKrK(_, m, k), mK_paged_cur_copy(_, ki));
                    }
                }
            }
        }
    };

    /**
     * 存储V张量数据到分页内存
     * 
     * @tparam TensorV V张量类型
     * @param n_block 块索引
     * @param tVrV 要存储的V张量数据
     */
    template <typename TensorV>
    CUTLASS_DEVICE
    void store_V(const int n_block, TensorV &&tVrV) {
        // 如果KV在同一迭代中处理，计算V指针
        if constexpr (KV_Same_Iter) { compute_V_ptr(); }
        // 仅用于索引计算，因为线程0的所有索引在编译时都是已知的
        auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
        Tensor cV = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        // 使用恒等布局重复分区
        Tensor tVcV = gmem_thr_copy_kv.partition_S(cV);
        Tensor t0VcV = gmem_thr0_copy_kv.partition_S(cV);

        GmemTiledCopyKVStore gmem_tiled_copy_kv_store;
        int const seqlenk_row_limit = std::min(seqlen_k - n_block * kBlockN, kBlockN) - get<0>(tVcV(_0{}, _0{}, _0{}));
        #pragma unroll
        for (int m = 0; m < size<1>(tVrV); ++m) {
            bool const should_load = get<0>(t0VcV(_0{}, m, _0{})) < seqlenk_row_limit;
            // 使用warp shuffle获取V指针
            Element* v_ptr = reinterpret_cast<Element*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrVPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
            Tensor mV_paged_cur = make_tensor(make_gmem_ptr(v_ptr), Shape<Int<kHeadDimV>>{});
            Tensor mV_paged_cur_copy = cute::tiled_divide(mV_paged_cur, Shape<Int<kGmemElemsPerLoad>>{});
            if (should_load) {
                #pragma unroll
                for (int k = 0; k < size<2>(tVrV); ++k) {
                    int const ki = get<1>(tVcV(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    // 只有在谓词为真时才存储
                    if (tVpV(_0{}, k)) {
                        cute::copy(gmem_tiled_copy_kv_store, tVrV(_, m, k), mV_paged_cur_copy(_, ki));
                    }
                }
            }
        }
        // 如果KV不在同一迭代中处理，为下次迭代计算V指针
        if constexpr (!KV_Same_Iter) { compute_V_ptr(); }
    };


};

} // namespace flash

/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "named_barrier.hpp"
#include "seqlen.h"
#include "block.h"
#include "mask.h"
#include "pack_gqa.h"
#include "paged_kv.h"
#include "rotary.h"
#include "utils.h"
#include "sm90_pipeline_no_cluster.hpp"

namespace flash {

using namespace cute;

template <int Stages, class ClusterShape_, class TileShape_MNK_, int kHeadDimV, class Element_, class ElementAccum_, class ArchTag_,
        bool Is_causal_, bool Is_local_, bool Has_softcap_, bool Varlen_, bool PagedKVNonTMA_, bool AppendKV_, bool HasQv_,
        bool MmaPV_is_RS, bool IntraWGOverlap, bool PackGQA_, bool Split_, bool V_colmajor_>
struct CollectiveMainloopFwdSm90 {

    static constexpr int kStages = Stages;
    using ClusterShape = ClusterShape_;
    using TileShape_MNK = TileShape_MNK_;
    using TileShape_MNK_PV = Shape<decltype(get<0>(TileShape_MNK{})), Int<kHeadDimV>, decltype(get<1>(TileShape_MNK{}))>;
    using TileShape_MNK_QV = Shape<decltype(get<0>(TileShape_MNK{})), decltype(get<1>(TileShape_MNK{})), Int<kHeadDimV>>;
    using Element = Element_;
    using ElementAccum = ElementAccum_;
    using ArchTag = ArchTag_;
    static constexpr bool Is_FP8 = cute::is_same_v<Element, cutlass::float_e4m3_t> || cute::is_same_v<Element, cutlass::float_e5m2_t>;;
    static constexpr bool Is_causal = Is_causal_;
    static constexpr bool Is_local = Is_local_;
    static constexpr bool Has_softcap = Has_softcap_;
    static constexpr bool Varlen = Varlen_;
    static constexpr bool PagedKVNonTMA = PagedKVNonTMA_;
    static constexpr bool AppendKV = AppendKV_;
    static constexpr bool HasQv = HasQv_;
    static constexpr bool PackGQA = PackGQA_;
    static constexpr bool Split = Split_;
    static constexpr bool V_colmajor = V_colmajor_;
    static constexpr bool Transpose_V = Is_FP8 && !V_colmajor;
    static constexpr bool Use_TMA_Q = !PackGQA;
    static constexpr bool Use_TMA_KV = !PagedKVNonTMA;
    static_assert(Use_TMA_KV || CUTE_STATIC_V(size(ClusterShape{})) == 1, "If not using TMA for KV, ClusterShape must be 1");
    static_assert(Use_TMA_KV || !V_colmajor, "If not using TMA for KV, V_colmajor is not supported");
    static constexpr bool SameHeadDim = get<2>(TileShape_MNK{}) == kHeadDimV;
    static constexpr bool LargeHeadDimV = kHeadDimV > 256;

    static_assert(ArchTag::kMinComputeCapability >= 90);

    static constexpr cute::GMMA::Major MmaMajorV = !Is_FP8 && !V_colmajor ? GMMA::Major::MN : GMMA::Major::K;
    static constexpr cute::GMMA::Major TmaMajorV = !V_colmajor ? GMMA::Major::MN : GMMA::Major::K;

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});

    using SeqlenInfo_t = flash::SeqlenInfoQKNewK<Varlen, AppendKV>;
    using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN, Is_causal, Is_local, PackGQA, Split>;

    static_assert(!LargeHeadDimV || kHeadDimV % 256 == 0);
    static_assert(!LargeHeadDimV || kBlockM <= 64, "kBlockM must be 64 or less for large Headdim_V");
    static_assert(!LargeHeadDimV || !MmaPV_is_RS, "MmaPV must be SS for large Headdim_V");

    // Register bandwidth is actually a bottleneck so we don't want Q to be in registers.
    // Leaving this option here for reference.
    static constexpr bool MmaQK_is_RS = false;
    // We can have MmaPV with P in smem in rmem to reduce register pressure at the cost of more smem.
    static_assert(!(!MmaPV_is_RS && Is_FP8), "MmaPV must be RS if FP8");
    static_assert(!(!MmaPV_is_RS && Transpose_V), "MmaPV must be RS if Transpose_V");

    // Slightly faster in this case to have WG1 use RS instead of SS to avoid waiting for the P smem write
    // true
    static constexpr bool MmaPV_use_RS_WG1 = !MmaPV_is_RS && kHeadDim == 64 && kHeadDimV == 512;

    using AtomLayoutQK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;
    using TiledMmaQK = decltype(cute::make_tiled_mma(
        std::conditional_t<
            !MmaQK_is_RS,
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>()),
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShape_MNK>())
        >{},
        AtomLayoutQK{}));
    using AtomLayoutPV = std::conditional_t<
        !LargeHeadDimV,
        AtomLayoutQK,
        Layout<Shape<_1, Int<kHeadDimV / 256>, _1>>
    >;
    using TiledMmaPV = decltype(cute::make_tiled_mma(
        std::conditional_t<
            !MmaPV_is_RS,
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum,
                     TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>()),
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum,
                     TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>())
        >{},
        AtomLayoutPV{}));
    using TiledMmaQV = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK_QV>(),
        AtomLayoutQK{}));
    // For hdim64,512, WG1 can use RS but WG2 must use SS
    using TiledMmaPV_RS = decltype(cute::make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>(),
        AtomLayoutPV{}));

    static constexpr int NumMmaThreadsQK = size(TiledMmaQK{});
    static constexpr int NumMmaThreads = size(TiledMmaPV{});
    static constexpr int NumProducerThreads = !Transpose_V && Use_TMA_KV && Use_TMA_Q ? cutlass::NumThreadsPerWarp : cutlass::NumThreadsPerWarpGroup;
    static_assert(NumMmaThreadsQK % cutlass::NumThreadsPerWarpGroup == 0);
    static_assert(NumMmaThreads % cutlass::NumThreadsPerWarpGroup == 0);
    static constexpr int NumMmaWarpGroups = NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
    static_assert(NumMmaWarpGroups == 1 || NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<TmaMajorV, Element,
                                      Int<kHeadDimV>, decltype(cute::get<2>(TileShape_MNK_PV{}))>());
    using SmemLayoutVt = decltype(tile_to_shape(
        SmemLayoutAtomVt{},
        make_shape(Int<kHeadDimV>{}, shape<2>(TileShape_MNK_PV{}), Int<kStages>{}),
        std::conditional_t<TmaMajorV == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

    using SmemLayoutAtomVtMma = decltype(cutlass::gemm::collective::detail::ss_smem_selector<MmaMajorV, Element,
                                         Int<kHeadDimV>, decltype(cute::get<2>(TileShape_MNK_PV{}))>());
    using SmemLayoutVtMma = decltype(tile_to_shape(
        SmemLayoutAtomVtMma{},
        make_shape(Int<kHeadDimV>{}, shape<2>(TileShape_MNK_PV{}), Int<kStages>{}),
        std::conditional_t<MmaMajorV == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

    using SmemLayoutAtomQv = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK_QV{})), decltype(cute::get<2>(TileShape_MNK_QV{}))>());
    using SmemLayoutQv = decltype(tile_to_shape(SmemLayoutAtomQv{}, select<0, 2>(TileShape_MNK_QV{})));
    using SmemLayoutAtomVMmaQV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK_QV{})), decltype(cute::get<2>(TileShape_MNK_QV{}))>());
    using SmemLayoutVMmaQV = decltype(tile_to_shape(
        SmemLayoutAtomVMmaQV{},
        make_shape(shape<1>(TileShape_MNK_QV{}), Int<kHeadDimV>{}, Int<kStages>{})));
    static_assert(CUTE_STATIC_V(size(SmemLayoutVMmaQV{})) == size(SmemLayoutVtMma{}));

    // Only used if we're using cp.async to load V
    using SmemLayoutAtomVCpAsync = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), Int<kHeadDimV>>());
    using SmemLayoutVCpAsync = decltype(tile_to_shape(
        SmemLayoutAtomVCpAsync{},
        make_shape(shape<1>(TileShape_MNK{}), Int<kHeadDimV>{}, Int<kStages>{})));

    using SmemLayoutAtomP = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutP = decltype(tile_to_shape(SmemLayoutAtomP{}, select<0, 1>(TileShape_MNK{})));

    // Only for LargeHeadDimV where WG0 sends WG1 the scales
    using SmemLayoutScale = cute::Layout<cute::Shape<Int<kBlockM>, Int<kStages>>>;

    using SmemCopyAtomP = Copy_Atom<cute::SM90_U32x4_STSM_N, Element>;

    // Use LDSM.T and STSM to transpose V in the case of FP8 and V being row-major.
    // For FP16/BF16 we don't do any transposing.
    static_assert(!Transpose_V || (kHeadDimV % 32 == 0 && kBlockN % 32 == 0));
    static constexpr bool kHeadDimV_multiple_64 = kHeadDimV % 64 == 0;
    // Either kHeadDimV is a multiple of 64 (in which case we use a block size of 64 x 32 for the transpose),
    // or we need kBlockN to be a multiple of 64 (in which case we use a block size of 32 x 64 for the transpose).
    static_assert(!Transpose_V || (kHeadDimV_multiple_64 || kBlockN % 64 == 0));
    using LDSM_thread_shape  = std::conditional_t<kHeadDimV_multiple_64, Shape<_32, _4, _1, _1>, Shape<_16, _4, _1, _2>>;
    using LDSM_thread_stride = std::conditional_t<kHeadDimV_multiple_64, Stride<_4, _1, _0, _0>, Stride<_4, _1, _0, _64>>;
    using LDSM_value_shape = Shape<_2, _2, _1, _4>;
    using LDSM_value_stride = Stride<_1, _2, _16, _4>;
    using LDSM_divide_shape = std::conditional_t<kHeadDimV_multiple_64, Shape<_64, _8>, Shape<_32, _8>>;
    using S2RTiledCopyVt = decltype(make_tiled_copy(
        Copy_Atom<SM75_U16x8_LDSM_T, Element>{}, Layout<LDSM_thread_shape, LDSM_thread_stride>{},
        Layout<LDSM_value_shape, LDSM_value_stride>{}));

    using STSM_thread_shape  = std::conditional_t<kHeadDimV_multiple_64, Shape<_8, _4, _4, _1>, Shape<_8, _4, _2, _2>>;
    using STSM_thread_stride = std::conditional_t<kHeadDimV_multiple_64, Stride<_4, _1, _32, _0>, Stride<_4, _1, _32, _64>>;
    using STSM_value_shape = Shape<_1, _4, _2, _2>;
    using STSM_value_stride = Stride<_0, _1, _4, _8>;
    using STSM_divide_shape = Shape<_8, _16>;
    // These will not permute the columns of V (the kHeadDimV dimension) but incur bank conflicts
    // so a little slower (e.g. 1150 TFLOPS for hdim 256 instead of 1200 TFLOPS).
    // Instead we will permute the cols of V, and un-permute the cols of O in the epilogue.
    // using STSM_value_shape = Shape<_2, _4, _1, _2>;
    // using STSM_value_stride = Stride<_4, _1, _0, _8>;
    // using STSM_divide_shape = Shape<_16, _16>;
    using R2STiledCopyV = decltype(make_tiled_copy(
        Copy_Atom<SM90_U32x4_STSM_N, Element>{}, Layout<STSM_thread_shape, STSM_thread_stride>{},
        Layout<STSM_value_shape, STSM_value_stride>{}));

    using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
    using GmemTiledCopyKV = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{})));

    // We use CpAsync for K and V if PagedKVNonTMA and AppendKV, since TMA doesn't work there
    static constexpr int kHeadDimGCD = cute::gcd(kHeadDim, kHeadDimV);
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDimGCD % kGmemElemsPerLoad == 0, "Headdim and HeaddimV must be a multiple of kGmemElemsPerLoad");
    // We want each "row" to have 64 elements (128 bytes, i.e. 1 cache line). E.g. if hdim=128, we want each
    // thread to have 4 loads in the M direction and 2 vectorized load in the K direction.
    // We want each thread to have at least 2 loads in the K direction since in the case of non-interleaved
    // rotary (combining elements at indices 0 and rotary_dim/2, 1 and rotary_dim/2+1, etc), each thread will
    // load twice from the same row.
    static constexpr int kBytePerHalfRow = kHeadDimGCD / 2 * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerHalfRow % 128 == 0 ? 128 : (kBytePerHalfRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(NumMmaThreads % kGmemThreadsPerRow == 0, "NumMmaThreads must be a multiple of kGmemThreadsPerRow");
    // We assume threads loading the same row are in the same warp. This is for an optimization in PagedKVNonTMA where
    // these threads share the same page table entry and share the work of computing pointers to paged K and paged V.
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0, "kGmemThreadsPerRow must divide NumThreadsPerWarp");
    using GmemLayoutAtom = Layout<Shape <Int<NumMmaThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    // If AppendKV, we'll be loading Q for rotary, and we assume divisibility to avoid predication
    static_assert(!AppendKV || kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0, "kBlockM must be a multiple of NumMmaThreads / kGmemThreadsPerRow");
    using GmemTiledCopyAppendKV = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store

    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQK = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using StrideV = std::conditional_t<!V_colmajor, StrideQK, cute::Stride<_1, int64_t, int64_t, int64_t>>;
    // ((qhead_per_khead, seqlen_q), d, nheads_kv, batch, num_splits)
    using ShapeQPacked = std::conditional_t<!PackGQA, ShapeQKV, cute::Shape<cute::Shape<int32_t, int32_t>, int32_t, int32_t, int32_t>>;
    using StrideQPacked = std::conditional_t<!PackGQA, StrideQK, cute::Stride<cute::Stride<int64_t, int64_t>, _1, int64_t, int64_t>>;
    using ShapePageTable = cute::Shape<int32_t, int32_t>;  // (batch, max_num_pages_per_seq)
    using StridePageTable = cute::Stride<int64_t, _1>;
    using ShapeRotary = cute::Shape<int32_t, int32_t>;  // (seqlen_ro, rotary_dim // 2)
    using StrideRotary = cute::Stride<int64_t, _1>;
    using StrideDescale = cute::Stride<int64_t, int64_t>;

    using TMA_Q = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyQ{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQK{}),
        SmemLayoutQ{},
        TileShape_MNK{},
        ClusterShape{}));

    using TMA_K = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQK{}),
        take<0, 2>(SmemLayoutK{}),
        TileShape_MNK{},
        ClusterShape{})); // mcast along M mode for this N load, if any

    using TMA_V = decltype(make_tma_copy(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, select<1, 0, 2, 3>(StrideV{})),
        take<0, 2>(SmemLayoutVt{}),
        select<1, 2>(TileShape_MNK_PV{}),
        size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any

    using TMA_Qv_ = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyQ{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQK{}),
        SmemLayoutQv{},
        TileShape_MNK_QV{},
        ClusterShape{}));
    using TMA_Qv = std::conditional_t<HasQv, TMA_Qv_, std::nullptr_t>;

    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutVt{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesQv = static_cast<uint32_t>(size(SmemLayoutQv{}) * cutlass::sizeof_bits_v<Element> / 8);

    using PipelineTmaAsync = std::conditional_t<CUTE_STATIC_V(size(ClusterShape{})) == 1, typename cutlass::PipelineTmaAsyncNoCluster<kStages>, typename cutlass::PipelineTmaAsync<kStages>>;
    using MainloopPipelineK = std::conditional_t<Use_TMA_KV, PipelineTmaAsync, typename cutlass::PipelineAsync<kStages>>;
    using MainloopPipelineV = std::conditional_t<!Transpose_V && Use_TMA_KV, PipelineTmaAsync, typename cutlass::PipelineAsync<kStages>>;
    using MainloopPipelineVt = std::conditional_t<Use_TMA_KV, PipelineTmaAsync, typename cutlass::PipelineAsync<kStages>>;
    // We always use TMA for K_new and V_new
    using MainloopPipelineKVNew = PipelineTmaAsync;
    using PipelineState = cutlass::PipelineState<kStages>;

    // If PackGQA, we use cp.async (instead of TMA) to load Q, so we want smem_q to be aligned
    // and have sQ being position_independent_swizzle_tensor.
    // If !Use_TMA_KV, we use cp.async (instead of TMA) to load K & V, so we want smem_k and smem_v to be aligned.
    static constexpr size_t SmemAlignmentQ = Use_TMA_Q && !MmaQK_is_RS ? 128 : cutlass::detail::alignment_for_swizzle(SmemLayoutQ{});
    static constexpr size_t SmemAlignmentK = Use_TMA_KV && !AppendKV ? 128 : cutlass::detail::alignment_for_swizzle(SmemLayoutK{});
    static constexpr size_t SmemAlignmentVtNoTranspose = cutlass::detail::alignment_for_swizzle(SmemLayoutVt{});
    static constexpr size_t SmemAlignmentQv = Use_TMA_Q ? 128 : cutlass::detail::alignment_for_swizzle(SmemLayoutQv{});
    static_assert(SmemAlignmentQ >= 128 and SmemAlignmentK >= 128 && SmemAlignmentVtNoTranspose >= 128, "Require at least 128B alignment");
    static constexpr size_t SmemAlignmentP = cutlass::detail::alignment_for_swizzle(SmemLayoutP{});
    static_assert(SmemAlignmentP >= 128, "Require at least 128B alignment");

    using SmemP_t = std::conditional_t<MmaPV_is_RS, cute::array<Element, 0>, cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>, SmemAlignmentP>>;
    using SmemScale_t = std::conditional_t<!LargeHeadDimV, cute::array<float, 0>, cute::array_aligned<float, cute::cosize_v<SmemLayoutScale>, 128>>;
    using SmemQv_t = std::conditional_t<!HasQv, cute::array<Element, 0>, cute::array_aligned<Element, cute::cosize_v<SmemLayoutQv>, SmemAlignmentQv>>;
    // Sometimes even with SmemP_t = cute::array<Element, 0>, putting it in the TensorStorage struct causes
    // smem size to go from 227KB to 228KB and we get "invalid argument".

    struct TensorStorageWithoutPNoTranspose : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentVtNoTranspose), _0> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, SmemAlignmentVtNoTranspose> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
        SmemQv_t smem_qv;
    };

    struct TensorStorageWithPNoTranspose : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentVtNoTranspose, SmemAlignmentP), _0> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, SmemAlignmentVtNoTranspose> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
        SmemQv_t smem_qv;
        SmemP_t smem_p;
    };
    struct TensorStorageWithPScaleNoTranspose : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentVtNoTranspose, SmemAlignmentP), _0> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, SmemAlignmentVtNoTranspose> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
        SmemQv_t smem_qv;
        SmemP_t smem_p;
        SmemScale_t smem_scale;
    };

    using TensorStorageNoTranspose = std::conditional_t<
        MmaPV_is_RS,
        TensorStorageWithoutPNoTranspose,
        std::conditional_t<!LargeHeadDimV, TensorStorageWithPNoTranspose, TensorStorageWithPScaleNoTranspose>
    >;

    static constexpr size_t SmemAlignmentVt = cutlass::detail::alignment_for_swizzle(SmemLayoutVt{});
    static constexpr size_t SmemAlignmentV = cutlass::detail::alignment_for_swizzle(SmemLayoutVtMma{});
    static_assert(SmemAlignmentVt >= 128 and SmemAlignmentV >= 128, "Require at least 128B alignment");
    struct TensorStorageTransposeV : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentV), _0> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVtMma>, SmemAlignmentV> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, SmemAlignmentVt> smem_vt;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
        SmemQv_t smem_qv;
        SmemScale_t smem_scale;
    };

    using TensorStorage = std::conditional_t<!Transpose_V, TensorStorageNoTranspose, TensorStorageTransposeV>;

    // These are tuned for speed. They don't affect correctness.
    static constexpr bool UseSchedulerBarrier = (IntraWGOverlap
        ? (NumMmaWarpGroups >= 2) && (!Is_FP8 ? kHeadDim <= 128 : kHeadDim >= 128)
        : NumMmaWarpGroups == 2)
        && !LargeHeadDimV;
    static constexpr bool RescaleOBeforeGemm = kHeadDim > 128 && (!Is_FP8 || V_colmajor) && IntraWGOverlap;

    // Host side kernel arguments
    struct Arguments {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQK const stride_Q;
        Element* const ptr_K;  // not Element const* since we might append to KV cache in-place
        ShapeQKV const shape_K;
        StrideQK const stride_K;
        Element* const ptr_V;
        int32_t const headdim_v;
        StrideV const stride_V;
        Element const* const ptr_K_new;
        ShapeQKV const shape_K_new;
        StrideQK const stride_K_new;
        Element const* const ptr_V_new;
        StrideV const stride_V_new;
        Element const* const ptr_Qv;
        StrideQK const stride_Qv;
        Element const* const ptr_rotary_cos;
        ShapeRotary const shape_rotary;
        StrideRotary const stride_rotary_cos;
        Element const* const ptr_rotary_sin;
        StrideRotary const stride_rotary_sin;
        bool const is_rotary_interleaved;
        int const* const ptr_pagetable;
        ShapePageTable const shape_pagetable;
        StridePageTable const stride_pagetable;
        float const softmax_scale;
        float const* ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
        StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
        int const window_size_left = -1, window_size_right = -1, attention_chunk = 0;
        float const softcap_val;
        int const num_splits;
        int const* const kv_batch_idx = nullptr;
        int const* const cu_seqlens_q = nullptr;
        int const* const cu_seqlens_k = nullptr;
        int const* const cu_seqlens_k_new = nullptr;
        int const* const seqused_q = nullptr;
        int const* const seqused_k = nullptr;
        int const* const leftpad_k = nullptr;
        int const* const seqlens_rotary = nullptr;
        int const mtp_step = 0;
    };

    // Device side kernel params
    struct Params {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQK const stride_Q;
        ShapeQPacked const shape_Q_packed;
        StrideQPacked const stride_Q_packed;
        Element* const ptr_K;
        ShapeQKV const shape_K;
        StrideQK const stride_K;
        Element* const ptr_V;
        int32_t const headdim_v;
        StrideV const stride_V;
        Element const* const ptr_K_new;
        ShapeQKV const shape_K_new;
        StrideQK const stride_K_new;
        Element const* const ptr_V_new;
        StrideV const stride_V_new;
        Element const* const ptr_Qv;
        StrideV const stride_Qv;
        ShapeQPacked const shape_Qv_packed;
        StrideQPacked const stride_Qv_packed;
        Element const* const ptr_rotary_cos;
        ShapeRotary const shape_rotary;
        StrideRotary const stride_rotary_cos;
        Element const* const ptr_rotary_sin;
        StrideRotary const stride_rotary_sin;
        bool const is_rotary_interleaved;
        int const* const ptr_pagetable;
        ShapePageTable const shape_pagetable;
        StridePageTable const stride_pagetable;
        cutlass::FastDivmod page_size_divmod;
        cutlass::FastDivmod blockN_per_page_size_divmod;
        cutlass::FastDivmod qhead_per_khead_divmod;
        TMA_Q tma_load_Q;
        TMA_K tma_load_K;
        TMA_V tma_load_V;
        TMA_K tma_load_K_new;
        TMA_V tma_load_V_new;
        TMA_Qv tma_load_Qv;
        float const softmax_scale_log2;
        float const* ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
        StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
        float const softcap_val;
        int const window_size_left, window_size_right;
        cutlass::FastDivmod attention_chunk_divmod;
        int const num_splits;
        int const* const kv_batch_idx = nullptr;
        int const* const cu_seqlens_q = nullptr;
        int const* const cu_seqlens_k = nullptr;
        int const* const cu_seqlens_k_new = nullptr;
        int const* const seqused_q = nullptr;
        int const* const seqused_k = nullptr;
        int const* const leftpad_k = nullptr;
        int const *const seqlens_rotary = nullptr;
        cutlass::FastDivmod qhead_per_khead_mtp_divmod;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
        TMA_Q tma_load_Q = make_tma_copy_A_sm90(
            GmemTiledCopyQ{},
            mQ,
            SmemLayoutQ{},
            TileShape_MNK{},
            ClusterShape{}); // no mcast for Q
        Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
        TMA_K tma_load_K = make_tma_copy_B_sm90(
            GmemTiledCopyKV{},
            mK,
            take<0, 2>(SmemLayoutK{}),
            TileShape_MNK{},
            ClusterShape{}); // mcast along M mode for this N load, if any
        Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V),
                                make_shape(args.headdim_v, get<0>(args.shape_K), get<2>(args.shape_K), get<3>(args.shape_K)),
                                select<1, 0, 2, 3>(args.stride_V));
        TMA_V tma_load_V = make_tma_copy(
            GmemTiledCopyKV{},
            mV,
            take<0, 2>(SmemLayoutVt{}),
            select<1, 2>(TileShape_MNK_PV{}),
            size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
        Tensor mKnew = make_tensor(make_gmem_ptr(args.ptr_K_new), args.shape_K_new, args.stride_K_new);
        TMA_K tma_load_K_new = make_tma_copy_B_sm90(
            GmemTiledCopyKV{},
            cute::conditional_return<AppendKV>(mKnew, mK),
            take<0, 2>(SmemLayoutK{}),
            TileShape_MNK{},
            ClusterShape{}); // mcast along M mode for this N load, if any
        Tensor mVnew = make_tensor(make_gmem_ptr(args.ptr_V_new),
                                   make_shape(args.headdim_v, get<0>(args.shape_K_new), get<2>(args.shape_K_new), get<3>(args.shape_K_new)),
                                   select<1, 0, 2, 3>(args.stride_V_new));
        TMA_V tma_load_V_new = make_tma_copy(
            GmemTiledCopyKV{},
            cute::conditional_return<AppendKV>(mVnew, mV),
            take<0, 2>(SmemLayoutVt{}),
            select<1, 2>(TileShape_MNK_PV{}),
            size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
        auto shape_Qv = make_shape(get<0>(args.shape_Q), args.headdim_v, get<2>(args.shape_Q), get<3>(args.shape_Q));
        Tensor mQv = make_tensor(make_gmem_ptr(args.ptr_Qv), shape_Qv, args.stride_Qv);
        TMA_Qv tma_load_Qv = [&] {
            if constexpr (HasQv) {
                return make_tma_copy_A_sm90(
                    GmemTiledCopyQ{},
                    mQv,
                    SmemLayoutQv{},
                    TileShape_MNK_QV{},
                    ClusterShape{}); // no mcast for Qv
            } else {
                return nullptr;
            }
        }();
        // If PackGQA, reshape Q to be ((qhead_per_khead, seqlen_q), head_size, nhead_k, batch_size)
        int const qhead_per_khead = !PackGQA ? 1 : cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K));
        auto const shape_Q_packed = cute::conditional_return<!PackGQA>(
            args.shape_Q,
            make_shape(make_shape(qhead_per_khead, get<0>(args.shape_Q)), get<1>(args.shape_Q), get<2>(args.shape_K), get<3>(args.shape_Q))
        );
        auto const stride_Q_packed = cute::conditional_return<!PackGQA>(
            args.stride_Q,
            make_stride(make_stride(get<2>(args.stride_Q), get<0>(args.stride_Q)), get<1>(args.stride_Q), get<2>(args.stride_Q) * qhead_per_khead, get<3>(args.stride_Q))
        );
        auto const shape_Qv_packed = cute::conditional_return<!PackGQA>(
            shape_Qv,
            make_shape(make_shape(qhead_per_khead, get<0>(shape_Qv)), get<1>(shape_Qv), get<2>(args.shape_K), get<3>(shape_Qv))
        );
        auto const stride_Qv_packed = cute::conditional_return<!PackGQA>(
            args.stride_Qv,
            make_stride(make_stride(get<2>(args.stride_Qv), get<0>(args.stride_Qv)), get<1>(args.stride_Qv), get<2>(args.stride_Qv) * qhead_per_khead, get<3>(args.stride_Qv))
        );
        if (get<1>(args.shape_rotary) > 0) {
            assert(args.ptr_rotary_cos != nullptr && args.ptr_rotary_sin != nullptr);
        }
        assert(args.num_splits >= 1);
        int page_size = !args.ptr_pagetable ? 1 : get<0>(args.shape_K);
        if (!PagedKVNonTMA && args.ptr_pagetable != nullptr) {
            assert(page_size % kBlockN == 0);
            assert(!args.leftpad_k);
        }
        // Avoid dividing by zero
        cutlass::FastDivmod attention_chunk_divmod(args.attention_chunk >= 1 ? args.attention_chunk : 1);
        attention_chunk_divmod.divisor = args.attention_chunk;
        
        cutlass::FastDivmod qhead_per_khead_divmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K)));

        cutlass::FastDivmod qhead_per_khead_mtp_divmod = qhead_per_khead_divmod;

        if (args.mtp_step > 0) {
            qhead_per_khead_mtp_divmod = cutlass::FastDivmod(cute::ceil_div(qhead_per_khead, args.mtp_step + 1));
        }
        // If there's tanh softcapping, we do tanh(scores * softmax_scale / softcap_val) * softcap_val.
        // Right after this, we multiply by log2(e) before applying exp2.
        // To reduce the number of instructions, we instead pre-multiply softmax_scale / softcap_val
        // (assigning it to params.softcap_val) and pre-multiply softcap_val * log2(e)
        // (assigning it to params.softmax_scale_log2).
        return {args.ptr_Q, args.shape_Q, args.stride_Q, shape_Q_packed, stride_Q_packed,
                args.ptr_K, args.shape_K, args.stride_K, args.ptr_V, args.headdim_v, args.stride_V,
                args.ptr_K_new, args.shape_K_new, args.stride_K_new, args.ptr_V_new, args.stride_V_new,
                args.ptr_Qv, args.stride_Qv, shape_Qv_packed, stride_Qv_packed,
                args.ptr_rotary_cos, args.shape_rotary, args.stride_rotary_cos,
                args.ptr_rotary_sin, args.stride_rotary_sin, args.is_rotary_interleaved,
                args.ptr_pagetable, args.shape_pagetable, args.stride_pagetable,
                cutlass::FastDivmod(page_size),  // page_size_divmod
                cutlass::FastDivmod(!args.ptr_pagetable ? 1 : cute::ceil_div(page_size, kBlockN)),  // blockN_per_page_size_divmod
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
                tma_load_Q, tma_load_K, tma_load_V, tma_load_K_new, tma_load_V_new, tma_load_Qv,
                !Has_softcap ? float(args.softmax_scale * M_LOG2E) : float(args.softcap_val * M_LOG2E),
                args.ptr_q_descale, args.ptr_k_descale, args.ptr_v_descale,
                args.stride_q_descale, args.stride_k_descale, args.stride_v_descale,
                !Has_softcap ? 0.f : args.softmax_scale / args.softcap_val,
                args.window_size_left, args.window_size_right, attention_chunk_divmod,
                !Split ? 1 : args.num_splits,
                args.kv_batch_idx,
                args.cu_seqlens_q, args.cu_seqlens_k, args.cu_seqlens_k_new,
                args.seqused_q, args.seqused_k, args.leftpad_k, args.seqlens_rotary,
                qhead_per_khead_mtp_divmod
            };
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        if constexpr (Use_TMA_Q) {
            cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
            if constexpr (HasQv) {
                cute::prefetch_tma_descriptor(params.tma_load_Qv.get_tma_descriptor());
            }
        }
        if constexpr (Use_TMA_KV) {
            cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
            cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
        }
        if constexpr (AppendKV) {
            cute::prefetch_tma_descriptor(params.tma_load_K_new.get_tma_descriptor());
            cute::prefetch_tma_descriptor(params.tma_load_V_new.get_tma_descriptor());
        }
    }

    template <typename SchedulerPrefetch, typename SharedStorage>
    CUTLASS_DEVICE void
    /**
     * Producer部分的数据加载函数 - Flash Attention前向传播的核心数据加载逻辑
     * 
     * 这个函数负责：
     * 1. 配置和管理Q、K、V张量的内存布局和数据加载
     * 2. 设置TMA(Tensor Memory Accelerator)加载器以高效访问显存
     * 3. 处理序列长度变化、因果掩码、滑动窗口等注意力机制变体
     * 4. 管理共享内存中的数据分区和流水线同步
     * 
     * @param params 包含所有计算参数的结构体，如张量形状、TMA描述符等
     * @param pipeline_k K张量的流水线管理器，用于异步数据加载
     * @param pipeline_v V张量的流水线管理器，用于异步数据加载  
     * @param pipeline_vt V转置张量的流水线管理器
     * @param smem_pipe_write 共享内存流水线的写入状态
     * @param shared_storage 共享内存存储结构，包含Q、K、V的共享内存缓冲区
     * @param scheduler_prefetch 调度器预取函数，用于优化内存访问
     * @param seqlen_info 序列长度信息，处理变长序列和偏移
     * @param block_coord 当前处理的块坐标 (m_block, bidh, bidb, split_idx)
     * @param work_idx 工作索引，用于任务调度
     */
    load(Params const& params,
         MainloopPipelineK pipeline_k,
         MainloopPipelineV pipeline_v,
         MainloopPipelineVt pipeline_vt,
         PipelineState& smem_pipe_write,
         SharedStorage &shared_storage,
         SchedulerPrefetch const& scheduler_prefetch,
         SeqlenInfo_t const& seqlen_info,
         cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
         int &work_idx
         ) {

        // 从块坐标中提取各维度信息 (这些变量在lambda中会被捕获，所以不能使用结构化绑定)
        int const m_block = get<0>(block_coord);     // M维度的块索引 (查询序列维度) 0
        int const bidh = get<1>(block_coord);        // 注意力头索引 0
        int const bidb = get<2>(block_coord);        // 批次索引 batch_id
        int const split_idx = get<3>(block_coord);   // 分片索引 (用于序列并行)
        
        // 计算当前块需要处理的N维度范围 (键值序列维度)
        // 根据因果掩码、滑动窗口、序列长度等约束条件确定有效的计算范围
        auto [n_block_min, n_block_max] = BlockMN_t::get_n_block_min_max(
            seqlen_info, m_block, bidb, split_idx, params.num_splits,
            params.window_size_left, params.window_size_right, params.attention_chunk_divmod,
            params.qhead_per_khead_divmod);
            
        // 检查是否存在有效的计算范围，避免非法内存访问
        // 在因果注意力、局部注意力、变长序列或分片计算中可能出现空范围
        if constexpr (Is_causal || Is_local || Varlen || Split) {
            if (n_block_max <= n_block_min) {
                scheduler_prefetch();  // 执行预取操作但跳过实际计算
                return;
            }
        }

        // =====================================================================
        // 共享内存张量配置阶段
        // =====================================================================
        // 为每个张量类型创建适当的共享内存视图，支持不同的数据布局和访问模式
        // 这些张量将作为全局内存和寄存器文件之间的缓冲区
        // =====================================================================
        
        // 为Q张量创建共享内存视图
        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        
        // 为K张量创建共享内存视图
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        
        // 创建位置无关的K张量视图，便于后续的LDSM和STSM转置操作中的地址计算
        Tensor sK_pi = as_position_independent_swizzle_tensor(sK);
        
        // as_position_independent_swizzle_tensor 简化了LDSM和STSM转置时的地址计算
        // 但要求smem_vt和smem_v按512字节等对齐
        
        // 根据是否需要转置V张量来配置V转置张量的共享内存布局
        Tensor sVt = [&] {
            if constexpr (!Transpose_V) {
                // 不转置时直接使用V的共享内存
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVt{});
            } else {
                // 需要转置时使用专门的Vt共享内存区域
                return cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_vt.data()), SmemLayoutVt{}));
            }
        }();
        
        // V张量的共享内存视图 (仅在需要转置V时使用)
        Tensor sV = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVtMma{}));
        
        // 用于cp.async异步加载V张量的共享内存视图
        Tensor sVcpasync = [&] {
            if constexpr (!Transpose_V) {
                // 不转置时使用V的共享内存区域
                return cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVCpAsync{}));
            } else {
                // 转置时使用Vt的共享内存区域
                return cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_vt.data()), SmemLayoutVCpAsync{}));
            }
        }();
        
        // Qv张量的共享内存视图 (用于某些Flash Attention变体)
        Tensor sQv = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_qv.data()), SmemLayoutQv{});

        // =====================================================================
        // 线程和批次索引计算
        // =====================================================================
        int const thread_idx = threadIdx.x % NumProducerThreads;  // 当前生产者线程索引
        // 计算K/V对应的注意力头索引 (支持分组查询注意力GQA) 0
        int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;
        // 计算K/V对应的批次索引 (支持批次索引重新映射) bidb
        int const bidb_kv = params.kv_batch_idx == nullptr ? bidb : params.kv_batch_idx[bidb];

        // =====================================================================
        // TMA加载器准备阶段
        // =====================================================================
        // TMA (Tensor Memory Accelerator) 是 H100 上的硬件加速内存传输单元
        // 支持多播、集群级别的数据传输优化
        // =====================================================================
        
        // 获取当前块在集群中的排名，用于TMA分区
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        // 计算集群内的局部块ID
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

        // 检查是否为变长序列
        bool const is_varlen_q = Varlen && params.cu_seqlens_q;  // Q序列是否变长
        bool const is_varlen_k = Varlen && params.cu_seqlens_k;  // K序列是否变长
        
        // =====================================================================
        // 全局内存张量配置阶段
        // =====================================================================
        // 配置Q张量的全局内存视图
        Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, bidh, !is_varlen_q ? bidb : 0);
        
        // 配置K张量的TMA全局内存视图
        Tensor mK_TMA = params.tma_load_K.get_tma_tensor(params.shape_K)(_, _, bidh_kv, _);
        
        // 配置V张量的形状并创建转置V的TMA全局内存视图
        auto shape_V = make_shape(params.headdim_v, get<0>(params.shape_K), get<2>(params.shape_K), get<3>(params.shape_K));
        Tensor mVt_TMA = params.tma_load_V.get_tma_tensor(shape_V)(_, _, bidh_kv, _);

        // =====================================================================
        // 局部分块配置阶段
        // =====================================================================
        // 为Q张量创建局部分块，考虑序列偏移
        Tensor gQ = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
        
        // 为K张量创建TMA局部分块
        Tensor gK_TMA = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}, _0{}), mK_TMA), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}, _));  // (N, K, _, _)
        
        // 为V转置张量创建TMA局部分块
        Tensor gVt_TMA = local_tile(domain_offset(make_coord(_0{}, seqlen_info.offset_k, _0{}), mVt_TMA), select<1, 2>(TileShape_MNK_PV{}), make_coord(_0{}, _, _));  // (K, N, _, _)

        // =====================================================================
        // TMA分区配置阶段
        // =====================================================================
        // Q张量的TMA分区
        auto block_tma_Q = params.tma_load_Q.get_slice(_0{});
        Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));  // 源端分区 (TMA)
        Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));  // 目标端分区 (TMA)
        
        // 如果使用TMA加载Q且为主线程，则进行预取
        if (Use_TMA_Q && thread_idx == 0) { prefetch(params.tma_load_Q, tQgQ); }
        
        // K张量的TMA分区 (手动处理position_independent_swizzle_tensor)
        auto block_tma_K = params.tma_load_K.get_slice(cluster_local_block_id.x);
        Tensor tKgK_TMA = group_modes<0, 3>(block_tma_K.partition_S(gK_TMA));  // (TMA, k, batch)
        Tensor tKsK_TMA = group_modes<0, 3>(block_tma_K.partition_D(sK));      // (TMA, PIPE)
        
        // V张量的TMA分区
        auto block_tma_V = params.tma_load_V.get_slice(cluster_local_block_id.x);
        Tensor tVgVt_TMA = group_modes<0, 3>(block_tma_V.partition_S(gVt_TMA)); // (TMA, k, batch)
        Tensor tVsVt_TMA = group_modes<0, 3>(block_tma_V.partition_D(sVt));     // (TMA, PIPE)
        
        // 如果支持Qv张量，配置其TMA分区
        auto [tQvgQv, tQvsQv] = [&] {
            if constexpr (HasQv) {
                // 构造Qv张量的形状
                auto shape_Qv = make_shape(get<0>(params.shape_Q), params.headdim_v, get<2>(params.shape_Q), get<3>(params.shape_Q));
                Tensor mQv = params.tma_load_Qv.get_tma_tensor(shape_Qv)(_, _, bidh, !is_varlen_q ? bidb : 0);
                Tensor gQv = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mQv), select<0, 2>(TileShape_MNK_QV{}), make_coord(m_block, _0{}));  // (M, Kv)
                auto block_tma_Qv = params.tma_load_Qv.get_slice(_0{});
                Tensor tQvgQv = group_modes<0, 3>(block_tma_Qv.partition_S(gQv));  // (TMA)
                Tensor tQvsQv = group_modes<0, 3>(block_tma_Qv.partition_D(sQv));  // (TMA)
                return cute::make_tuple(tQvgQv, tQvsQv);
            } else {
                return cute::make_tuple(nullptr, nullptr);
            }
        }();

        // This is used to index into the batch dimension of mK and mV 0
        int const bidb_kv_idx = !is_varlen_k && !params.ptr_pagetable ? bidb_kv : 0;

        // =====================================================================
        // PagedKV 管理器初始化
        // =====================================================================
        // PagedKV 支持动态页表管理，允许非连续的 K/V 存储
        // 这对于长序列和内存优化场景特别有用
        // =====================================================================
        // <64, 64, 64, 128, float16, true>
        using PagedKVManager_t = PagedKVManager<get<1>(TileShape_MNK{}), get<2>(TileShape_MNK{}), get<1>(TileShape_MNK_PV{}), NumProducerThreads, Element, Transpose_V || !IntraWGOverlap /*KV_Same_Iter*/>;
        PagedKVManager_t paged_kv_manager(
            params.ptr_pagetable, params.shape_pagetable, params.stride_pagetable,
            params.ptr_K, params.shape_K, params.stride_K,
            params.ptr_V, params.headdim_v, params.stride_V,
            params.page_size_divmod, params.blockN_per_page_size_divmod,
            bidb_kv, bidh_kv, thread_idx, seqlen_info.seqlen_k, seqlen_info.leftpad_k, bidb_kv_idx
        );

        // =====================================================================
        // V 张量转置设置 (仅在 Transpose_V 为 true 时使用)
        // =====================================================================
        // V 转置优化：使用 LDSM 和 STSM 指令进行高效的矩阵转置
        // 这个转置操作需要在生产者和消费者之间精确同步
        // =====================================================================
        S2RTiledCopyVt s2r_tiled_copy_vt;
        R2STiledCopyV r2s_tiled_copy_v;
        // 为当前线程创建V转置的共享内存→寄存器拷贝操作对象
        // s2r = Shared memory to Register，用于从共享内存读取Vt数据到寄存器
        // 每个线程处理Vt张量的一个特定切片，由thread_idx确定
        auto s2r_thr_copy_vt = s2r_tiled_copy_vt.get_thread_slice(thread_idx);
        
        // 为当前线程创建V的寄存器→共享内存拷贝操作对象  
        // r2s = Register to Shared memory，用于将转置后的V数据从寄存器写回共享内存
        // 配合LDSM.T和STSM指令实现高效的就地矩阵转置
        auto r2s_thr_copy_v = r2s_tiled_copy_v.get_thread_slice(thread_idx);
        // flat_divide(sVt, LDSM_divide_shape{}):  (64, 8, kHeadDim / 64, kBlockN / 8, kStages)
        Tensor tTranssVt_ = s2r_thr_copy_vt.partition_S(flat_divide(sVt, LDSM_divide_shape{}));  // ((16, 1), 1, 1, kHeadDim / 64, kBlockN / 32, kStages)
        // flat_divide(sV, STSM_divide_shape{}):  (8, 16, kHeadDim / 8, (4, kBlockN / 64), kStages)
        Tensor tTranssV_ = r2s_thr_copy_v.partition_D(flat_divide(sV, STSM_divide_shape{}));  // ((16, 1), 1, 1, kHeadDim / 64, (2, kBlockN / 64), kStages)
        CUTE_STATIC_ASSERT_V(rank(tTranssVt_) == rank(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<0>(tTranssVt_) == size<0>(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<1>(tTranssVt_) == size<1>(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<2>(tTranssVt_) == size<2>(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<3>(tTranssVt_) == size<3>(tTranssV_));
        CUTE_STATIC_ASSERT_V(size<4>(tTranssVt_) == size<4>(tTranssV_));
        // Faster to have 2 LDSM.T, byte permute, STSM for better ILP
        static constexpr int Transpose_ILP = (size<2>(tTranssVt_) * size<3>(tTranssVt_)) % 2 == 0 ? 2 : 1;
        Tensor tTranssVt = logical_divide(group_modes<1, rank(tTranssVt_) - 1>(tTranssVt_), Shape<Underscore, Int<Transpose_ILP>>{});  // ((16, 1), (2, kHeadDim / 64 * kBlockN / 32 / 2), kStages)
        Tensor tTranssV = logical_divide(group_modes<1, rank(tTranssV_) - 1>(tTranssV_), Shape<Underscore, Int<Transpose_ILP>>{});  // ((16, 1), (2, kHeadDim / 64 * kBlockN / 32 / 2), kStages)
        
        // =====================================================================
        // V 张量转置操作 Lambda 函数
        // =====================================================================
        // 使用字节置换优化的转置操作，提高指令级并行度 (ILP)
        // =====================================================================
        auto transpose_V = [&](int stage) {
            if constexpr (Transpose_V) {
                #pragma unroll
                for (int i = 0; i < size<1, 1>(tTranssVt); ++i) {
                    Tensor tTransrV = make_fragment_like(tTranssV(_, make_coord(_, _0{}), _0{}));
                    static_assert(size<0>(tTransrV) == 16);
                    Tensor tTransrV_64 = recast<uint2>(tTransrV);
                    cute::copy(s2r_tiled_copy_vt, tTranssVt(_, make_coord(_, i), stage), tTransrV);
                    #pragma unroll
                    for (int j = 0; j < size(tTransrV_64); ++j) {
                        uint32_t upper = tTransrV_64[j].x;
                        uint32_t lower = tTransrV_64[j].y;
                        tTransrV_64[j].x = __byte_perm(upper, lower, 0x6420);
                        tTransrV_64[j].y = __byte_perm(upper, lower, 0x7531);
                    }
                    cute::copy(r2s_tiled_copy_v, tTransrV, tTranssV(_, make_coord(_, i), stage));
                }
            }
        };

        // =====================================================================
        // TMA 多播掩码设置
        // =====================================================================
        // 配置集群内的多播模式，允许单次 TMA 操作向多个 SM 广播数据
        // =====================================================================
        
        // 初始化K/V数据的多播掩码，16位掩码对应最多16个CTA块
        uint16_t mcast_mask_kv = 0;
        
        // 仅在使用TMA多播加载时才需要配置掩码
        if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
            // 创建集群布局：将2D集群坐标(m,n)映射到1D块ID
            auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
            
            // 遍历M维度(行方向)的所有块
            // 目标：为同一N列(cluster_local_block_id.y)的所有M行块设置多播位
            for (int m = 0; m < size<0>(block_layout); ++m) {
                // 计算位置(m, 当前块的y坐标)对应的块ID
                // 在多播掩码中设置对应位，表示该块应接收K/V数据
                // 这样一次TMA操作可以将K/V数据同时传输给同一列的所有块
                mcast_mask_kv |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
            }
        }

        // =====================================================================
        // K 张量加载 Lambda 函数
        // =====================================================================
        // 支持 TMA 和非 TMA 两种加载模式，根据是否需要序列长度掩码进行优化
        // =====================================================================
        auto load_K = [&] (int const n_block, auto const& smem_pipe_write, auto need_seqlenk_masking_type) {
            pipeline_k.producer_acquire(smem_pipe_write);
            if constexpr (!PagedKVNonTMA) {
                auto [n_block_idx, bidb_kv_idx] = paged_kv_manager.get_indices_for_K_TMA();
                copy(params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
                    tKgK_TMA(_, n_block_idx, bidb_kv_idx), tKsK_TMA(_, smem_pipe_write.index()));
            } else {
                constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
                paged_kv_manager.template load_K<Seqlenk_mask>(n_block, sK_pi(_, _, smem_pipe_write.index()));
                pipeline_k.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);
            }
        };

        // =====================================================================
        // V 张量加载 Lambda 函数
        // =====================================================================
        auto load_V = [&] (int const n_block, auto const& smem_pipe_write, auto need_seqlenk_masking_type) {
            auto pipeline_v_load = cute::conditional_return<!Transpose_V>(pipeline_v, pipeline_vt);
            pipeline_v_load.producer_acquire(smem_pipe_write);
            if constexpr (!PagedKVNonTMA) {
                auto [n_block_idx, bidb_kv_idx] = paged_kv_manager.get_indices_for_V_TMA();
                copy(params.tma_load_V.with(*pipeline_v_load.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
                    tVgVt_TMA(_, n_block_idx, bidb_kv_idx), tVsVt_TMA(_, smem_pipe_write.index()));
            } else {
                constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
                paged_kv_manager.template load_V<Seqlenk_mask>(n_block, sVcpasync(_, _, smem_pipe_write.index()));
                pipeline_v_load.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);
            }
        };

        // =====================================================================
        // V 转置拷贝操作 Lambda 函数
        // =====================================================================
        // 【关键同步点】TransposeBarrier 协调
        // 
        // 目的：确保 V 转置操作的生产者-消费者同步
        // 流程：
        // 1. 等待 Vt 数据在共享内存中可用 (pipeline_vt.consumer_wait)
        // 2. 获取 V 流水线的写入权限 (pipeline_v.producer_acquire)
        // 3. 执行转置操作 (transpose_V)
        // 4. 内存栅栏确保转置完成 (fence_view_async_shared)
        // 5. 提交 V 流水线 (pipeline_v.producer_commit)
        // 6. TransposeBarrier 同步确保 warpgroup 间协调
        // 7. 释放 Vt 流水线 (pipeline_vt.consumer_release)
        // =====================================================================
        auto copy_Vt_to_V = [&] (auto const& smem_pipe_write) {
            // Instead of maintaining smem_pipe_read as a separate variable, we can just use smem_pipe_write,
            // and exploit the invariance that smem_pipe_write.phase() == smem_pipe_read.phase() ^ 1.
            // This saves 1 or 2 registers.
            PipelineState smem_pipe_read{smem_pipe_write.index(), smem_pipe_write.phase() ^ 1, smem_pipe_write.count()};
            pipeline_vt.consumer_wait(smem_pipe_read);
            pipeline_v.producer_acquire(smem_pipe_write);
            transpose_V(smem_pipe_write.index());
            // SMEM fence to make sure V is transposed before math
            cutlass::arch::fence_view_async_shared();
            pipeline_v.producer_commit(smem_pipe_write);
            // ========================================================
            // 【关键同步点 3】TransposeBarrier
            // ========================================================
            // 目的：确保 V 转置操作中 warpgroup 间的同步
            // 参与者：NumThreadsPerWarpGroup (单个 warpgroup 的所有线程)
            // 机制：Named Barrier 确保转置操作的原子性
            // 
            // 重要性：PipelineTmaAsync::consumer_release 要求 warpgroup 
            // 在调用前已同步，否则会出现竞态条件
            // ========================================================
            cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, cutlass::arch::ReservedNamedBarriers::TransposeBarrier /*id*/);
            pipeline_vt.consumer_release(smem_pipe_read);
        };

        int n_block = n_block_max - 1;

        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        // If this is true, we're guaranteed that only the first warp will execute this function
        // 128 != 32
        static constexpr bool SingleProducerWarp = NumProducerThreads == cutlass::NumThreadsPerWarp;
        bool should_load_KV = !Use_TMA_KV || ((SingleProducerWarp || warp_idx_in_warpgroup == 0) && cute::elect_one_sync());

        // =====================================================================
        // 初始 K/V 加载阶段
        // =====================================================================
        if (should_load_KV) {
            if constexpr (PagedKVNonTMA) {
                paged_kv_manager.template load_page_table<true /*Seqlenk_mask*/, true /*First_iter*/>(n_block);
            } else {
                paged_kv_manager.template load_page_table_TMA<true /*First_iter*/>(n_block);
            }
            if constexpr (Transpose_V) { load_V(n_block, smem_pipe_write, cute::true_type{} /*Seqlenk_mask*/); }
            // if (thread_idx == 0) { printf("Producer: main load, before load_K, index = %d\n", smem_pipe_write.index());}
            load_K(n_block, smem_pipe_write, cute::true_type{} /*Seqlenk_mask*/);
            // if (thread_idx == 0) { printf("Producer: main load, after load K, index = %d\n", smem_pipe_write.index());}
        }

        // =====================================================================
        // Q 张量加载阶段
        // =====================================================================
        // 支持 TMA 和 cp.async 两种加载模式
        // 需要与消费者通过 QueryEmpty barrier 进行同步
        // =====================================================================
        if constexpr (Use_TMA_Q) {
            // ========================================================
            // 【关键同步点 4】QueryEmpty Barrier (TMA 模式)
            // ========================================================
            // 目的：确保 Q 共享内存区域可以安全写入
            // 参与者：NumMmaThreadsQK + NumThreadsPerWarp (消费者 + 生产者第一个 warp)
            // 机制：Named Barrier - 等待消费者释放 Q 共享内存使用权
            // 
            // 时序关系：
            // 1. 消费者完成 Q@K^T 计算后释放 Q 区域
            // 2. 生产者在此等待，确保可以安全覆盖 Q 数据
            // 3. 生产者开始 TMA 加载新的 Q 数据
            // ========================================================
            // Wait for the MMA warpgroups to signal that smem_q is ready
            if (SingleProducerWarp || warp_idx_in_warpgroup == 0) {
                cutlass::arch::NamedBarrier::sync(NumMmaThreadsQK + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);
            }

            if ((SingleProducerWarp || warp_idx_in_warpgroup == 0) && cute::elect_one_sync()) {
                shared_storage.pipelines.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
                copy(params.tma_load_Q.with(reinterpret_cast<typename cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_Q), 0 /*mcast_mask*/, !Split ? TMA::CacheHintSm90::EVICT_FIRST : TMA::CacheHintSm90::EVICT_LAST),
                    tQgQ, tQsQ);
                if constexpr (HasQv) {
                    shared_storage.pipelines.barrier_Qv.arrive_and_expect_tx(TmaTransactionBytesQv);
                    copy(params.tma_load_Qv.with(reinterpret_cast<typename cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_Qv), 0 /*mcast_mask*/, !Split ? TMA::CacheHintSm90::EVICT_FIRST : TMA::CacheHintSm90::EVICT_LAST),
                        tQvgQv, tQvsQv);
                }
            }
        } else {  // Load Q with cp.async
            // ========================================================
            // 【关键同步点 5】QueryEmpty Barrier (cp.async 模式)
            // ========================================================
            // 与 TMA 模式类似，但参与线程数不同
            // 参与者：NumMmaThreadsQK + NumProducerThreads (消费者 + 所有生产者)
            // ========================================================
            cutlass::arch::NamedBarrier::sync(NumMmaThreadsQK + NumProducerThreads, static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);
            Tensor mQ = make_tensor(make_gmem_ptr(params.ptr_Q + seqlen_info.offset_q * get<0>(params.stride_Q)), params.shape_Q_packed, params.stride_Q_packed)(_, _, bidh, !is_varlen_q ? bidb : 0);
            Tensor sQ_pi = cute::as_position_independent_swizzle_tensor(sQ);
            using PackGQAt = flash::PackGQAManager<get<0>(TileShape_MNK{}), get<2>(TileShape_MNK{}), NumProducerThreads, Element>;
            PackGQAt::load_Q(mQ, sQ_pi, params.qhead_per_khead_divmod, thread_idx, seqlen_info.seqlen_q, m_block);
            auto &barrier_Q = shared_storage.pipelines.barrier_Q;
            cutlass::arch::cpasync_barrier_arrive(reinterpret_cast<uint64_t*>(&barrier_Q));
            barrier_Q.arrive();
            if constexpr (HasQv) {
                Tensor mQv = make_tensor(make_gmem_ptr(params.ptr_Qv + seqlen_info.offset_q * get<0>(params.stride_Qv)), params.shape_Qv_packed, params.stride_Qv_packed)(_, _, bidh, !is_varlen_q ? bidb : 0);
                Tensor sQv_pi = cute::as_position_independent_swizzle_tensor(sQv);
                using PackGQAt = flash::PackGQAManager<get<0>(TileShape_MNK_QV{}), get<2>(TileShape_MNK_QV{}), NumProducerThreads, Element>;
                PackGQAt::load_Q(mQv, sQv_pi, params.qhead_per_khead_divmod, thread_idx, seqlen_info.seqlen_q, m_block);
                auto &barrier_Qv = shared_storage.pipelines.barrier_Qv;
                cutlass::arch::cpasync_barrier_arrive(reinterpret_cast<uint64_t*>(&barrier_Qv));
                barrier_Qv.arrive();
            }
        }

        // ========================================================
        // 【关键同步点 6】barrier_O - 输出屏障
        // ========================================================
        // 目的：确保消费者完成输出张量写入，V 共享内存区域可以重用
        // 参与者：生产者和消费者通过 ClusterBarrier 协调
        // 机制：ClusterBarrier (不是 NamedBarrier) 确保跨 CTA 同步
        // 
        // 重要性：
        // - 防止生产者过早覆盖 V 数据
        // - 确保 TMA 多播操作的正确性
        // - 协调集群内不同 CTA 的执行时序
        // 
        // 时序关系：
        // 1. 消费者完成 Attention@V 计算和输出写入
        // 2. 消费者信号 barrier_O
        // 3. 生产者在此等待
        // 4. 确认输出完成后，生产者可以安全加载新的 V 数据
        // ========================================================
        // Wait for the MMA WGs to signal that smem_v are ready and V can be copied from gmem
        // Need ClusterBarrier, not just NamedBarrier. Otherwise we might have CTA 0 finishing the
        // TMA store on O first, call TMA multicast load on V, before CTA 1 can finishing TMA store on O.
        // if (thread_idx == 0) { printf("Producer: main load, before barrier_O, work_idx = %d\n", work_idx);}
        shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);
        // if (thread_idx == 0) { printf("Producer: main load, after barrier_O\n");}

        if constexpr (!Transpose_V && !IntraWGOverlap) {
            if (should_load_KV) { load_V(n_block, smem_pipe_write, cute::true_type{} /*Seqlenk_mask*/); }
        }
        int n_block_prev = n_block;
        --n_block;
        
        // =====================================================================
        // 主加载循环
        // =====================================================================
        // 遍历所有 N 维度块，按倒序加载以支持因果注意力
        // 实现流水线化的 K/V 加载和可选的 V 转置操作
        // =====================================================================
        #pragma unroll (!Transpose_V && Use_TMA_KV ? 2 : 1)
        for (; n_block >= n_block_min; --n_block) {
            PipelineState smem_pipe_write_v = smem_pipe_write; // copy the state, write_v is always 1 step behind
            ++smem_pipe_write;
            if (should_load_KV) {
                if constexpr (PagedKVNonTMA) {
                    paged_kv_manager.template load_page_table<false /*Seqlenk_mask*/>(n_block);
                } else {
                    paged_kv_manager.load_page_table_TMA(n_block);
                }
                if constexpr (Transpose_V) { load_V(n_block, smem_pipe_write, cute::false_type{} /*Seqlenk_mask*/); }
                load_K(n_block, smem_pipe_write, cute::false_type{} /*Seqlenk_mask*/);
                if constexpr (!Transpose_V) {
                    if constexpr (IntraWGOverlap) {
                        load_V(n_block_prev, smem_pipe_write_v, cute::true_type{} /*Seqlenk_mask*/);
                    } else {
                        load_V(n_block, smem_pipe_write, cute::false_type{} /*Seqlenk_mask*/);
                    }
                }
            }
            n_block_prev = n_block;
            if constexpr (Transpose_V) { copy_Vt_to_V(smem_pipe_write_v); }
        }
        scheduler_prefetch();
        if constexpr (!Transpose_V && IntraWGOverlap) {
            if (should_load_KV) { load_V(n_block_prev, smem_pipe_write, cute::true_type{} /*Seqlenk_mask*/); }
        }
        if constexpr (Transpose_V) { copy_Vt_to_V(smem_pipe_write); }
        ++smem_pipe_write;
        // At the end, all threads have the correct smem_pipe_write.
        ++work_idx;
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void
    load_tail(MainloopPipelineK pipeline_k, MainloopPipelineV pipeline_v, MainloopPipelineVt pipeline_vt,
              PipelineState& smem_pipe_write, SharedStorage &shared_storage, int const work_idx) {
        // If we don't wait for barrier_O here, when using Cluster, CTA0 might exit early and CTA1 will
        // try to arrive on barrier_O of CTA0, causing "unspecified launch failure".
        shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        // Issue the epilogue waits
        // TODO: check if this should be called by 1 thread or more
        if (warp_idx_in_warpgroup == 0 && cute::elect_one_sync()) {
            /* This helps avoid early exit of blocks in Cluster
            *  Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
            *  then would just be acquired since the phase was still inverted from make_producer_start_state
            */
            pipeline_k.producer_tail(smem_pipe_write);
            pipeline_v.producer_tail(smem_pipe_write);
            if constexpr (Transpose_V) { pipeline_vt.producer_tail(smem_pipe_write); }
        }
    }

    CUTLASS_DEVICE void
    warp_scheduler_barrier_sync() {
        if constexpr (UseSchedulerBarrier) {
            cutlass::arch::NamedBarrier::sync(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + flash::canonical_warp_group_idx_nosync() /*id*/);
        }
    }

    CUTLASS_DEVICE void
    warp_scheduler_barrier_arrive() {
        if constexpr (UseSchedulerBarrier) {
            static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);
            int const cur_WG = flash::canonical_warp_group_idx_nosync() - 1;
            int const next_WG = NumMmaWarpGroups == 2
                ? 1 - cur_WG
                : (cur_WG < NumMmaWarpGroups - 1 ? cur_WG + 1 : 0);
            cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) + next_WG /*id*/);
        }
    }

    CUTLASS_DEVICE void
    mma_init() {
        int warp_group_idx = flash::canonical_warp_group_idx_nosync();
        // Tell producers that smem_q is ready
        if (!LargeHeadDimV || warp_group_idx == 1) {
            cutlass::arch::NamedBarrier::arrive(NumMmaThreadsQK + (Use_TMA_Q ? cutlass::NumThreadsPerWarp : NumProducerThreads), static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);
        }
        if (LargeHeadDimV && warp_group_idx > 1) {
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PEmpty) /*id*/);
        }
        if constexpr (UseSchedulerBarrier) {
            // We have NamedBarrier for up to 3 WGs
            static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);
            // WG1 needs the very first signal to start
            if (warp_group_idx == 1) {
                cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) /*id*/);
            }
        }
    }

    template <typename SharedStorage, typename FrgTensorO, typename Softmax>
    CUTLASS_DEVICE bool
    /**
     * MMA (矩阵乘法累加) 函数 - Flash Attention 算法的核心计算函数
     * 
     * 该函数实现了 Flash Attention 的主要计算流程：
     * 1. Q * K^T 矩阵乘法计算attention scores
     * 2. 应用softmax归一化和在线更新算法
     * 3. P * V 矩阵乘法计算最终输出
     * 
     * @param params - 包含所有计算参数的结构体
     * @param pipeline_k - K矩阵的pipeline状态管理
     * @param pipeline_v - V矩阵的pipeline状态管理  
     * @param smem_pipe_read - 共享内存pipeline读取状态
     * @param tOrO - 输出张量O的fragment表示
     * @param softmax - softmax计算和在线更新的管理器
     * @param thread_idx - 当前线程索引
     * @param work_idx - 工作索引，用于barrier同步
     * @param seqlen_info - 序列长度信息
     * @param block_coord - 块坐标 (m_block, head_idx, batch_idx, split_idx)
     * @param shared_storage - 共享内存存储区域
     */
    mma(Params const& params,
        MainloopPipelineK pipeline_k,
        MainloopPipelineV pipeline_v,
        PipelineState& smem_pipe_read,
        FrgTensorO& tOrO,
        Softmax& softmax,
        int const thread_idx,
        int &work_idx,
        SeqlenInfo_t const& seqlen_info,
        cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
        SharedStorage& shared_storage
        ) {
        // 静态断言：确保输出张量O驻留在寄存器内存中以获得最佳性能
        static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");
        
        // 提取tile形状常量：块的M和N维度大小
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});

        // 从block_coord中提取坐标信息（不能使用结构化绑定，因为lambda无法捕获）
        int const m_block = get<0>(block_coord);      // 当前处理的M维度块索引 batch_id
        int const bidh = get<1>(block_coord);         // 注意力头索引 0
        int const bidb = get<2>(block_coord);         // 批次索引 batch_id
        int const split_idx = get<3>(block_coord);    // 分割索引（用于长序列处理）0
        
        // 计算KV头索引（用于分组查询注意力GQA） 0
        int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;
        
        // 计算当前块需要处理的N维度范围 [n_block_min, n_block_max)
        // 考虑因果掩码、局部注意力窗口、序列长度等约束
        // (0, 8446 // 64 = 132)
        auto [n_block_min, n_block_max] = BlockMN_t::get_n_block_min_max(
            seqlen_info, m_block, bidb, split_idx, params.num_splits,
            params.window_size_left, params.window_size_right, params.attention_chunk_divmod,
            params.qhead_per_khead_divmod);
            
        // 早期退出检查：如果没有有效的块需要处理，直接返回
        if constexpr (Is_causal || Is_local || Varlen || Split) {
            if (n_block_max <= n_block_min) { return false; }
        }

        // === 创建共享内存张量视图 ===
        // 为Q、K、V、P矩阵创建共享内存张量，使用相应的布局
        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVtMma{});
        
        // sP张量：根据MmaPV_is_RS标志决定是否复用smem_q的空间
        Tensor sP = [&] {
            if constexpr (MmaPV_is_RS) {
                // 如果P*V使用寄存器存储，可以复用Q的共享内存空间
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutP{});
            } else {
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutP{});
            }
        }();
        
        // sScale张量：仅在大头维度V时使用，用于存储softmax缩放因子
        Tensor sScale = [&] {
            if constexpr (LargeHeadDimV) {
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_scale.data()), SmemLayoutScale{});
            } else { 
                // 占位符，实际不会使用
                return make_tensor(make_smem_ptr(static_cast<float*>(nullptr)), SmemLayoutScale{});
            }
        }();
        
        // 用于Q*V计算的额外张量（特定场景下使用）
        Tensor sQv = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_qv.data()), SmemLayoutQv{});
        Tensor sVMmaQV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVMmaQV{});

        // === 验证MMA布局约束 ===
        if constexpr (!MmaQK_is_RS) {
            // 确保MMA布局满足warp group的要求
            static_assert(stride<0>(typename TiledMmaQK::ALayout{}) == 0 and
                        stride<0>(typename TiledMmaQK::BLayout{}) == 0 and
                        size<0>(typename TiledMmaQK::ALayout{}) == cutlass::NumThreadsPerWarpGroup and
                        size<0>(typename TiledMmaQK::BLayout{}) == cutlass::NumThreadsPerWarpGroup,
                "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
        }
        
        // === 配置warp group和MMA操作 ===
        // 256 // 128 = 2
        static constexpr int MmaWarpGroups = size(TiledMmaPV{}) / cutlass::NumThreadsPerWarpGroup;
        Layout warp_group_thread_layout = make_layout(make_shape(Int<MmaWarpGroups>{}),
                                                      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

        // 计算当前线程所属的warp group索引
        // warpgroup 1: 执行 Q@K^T 和 Softmax
        // warpgroup 2: 执行 Attention@V
        int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
        
        // 创建不同类型的MMA操作对象
        TiledMmaQK tiled_mma_qk;  // Q*K矩阵乘法
        TiledMmaPV tiled_mma_pv;  // P*V矩阵乘法  
        TiledMmaQV tiled_mma_qv;  // Q*V矩阵乘法（特定场景）
        
        // 为当前warp group获取对应的MMA切片
        auto wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_qv = tiled_mma_qv.get_slice(warp_group_thread_layout(warp_group_idx));

        // === 配置共享内存拷贝操作 ===
        auto smem_tiled_copy_P = make_tiled_copy_C(SmemCopyAtomP{}, tiled_mma_qk);
        auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(thread_idx);

        // === 分配fragment（寄存器级别的数据块）===
        Tensor tSrQ = wg_mma_qk.partition_fragment_A(sQ);   // Q矩阵fragment
        Tensor tSrK = wg_mma_qk.partition_fragment_B(sK);   // K矩阵fragment
        Tensor tOrV = wg_mma_pv.partition_fragment_B(sV);   // V矩阵fragment
        Tensor tOsP = wg_mma_pv.partition_fragment_A(sP);   // P矩阵fragment
        Tensor tSrQv = wg_mma_qv.partition_fragment_A(sQv); // Qv矩阵fragment
        Tensor tSrV = wg_mma_qv.partition_fragment_B(sVMmaQV); // V矩阵fragment（用于Q*V）
        
        // P矩阵的共享内存拷贝fragment
        Tensor tPsP = smem_thr_copy_P.partition_D(cute::as_position_independent_swizzle_tensor(sP));

        // === 配置缩放因子存储（仅在大头维度V时使用）===
        auto thread_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);
        Tensor taccOcO = thread_mma_pv.partition_C(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));
        Tensor taccOcO_rowcol = make_tensor(taccOcO.data(), flash::convert_layout_acc_rowcol(taccOcO.layout()));
        Tensor taccOcO_row = taccOcO_rowcol(_, _0{});
        
        // Lambda函数：将softmax缩放因子存储到共享内存
        auto store_scales = [&](auto& scales, int stage) {
            static_assert(CUTE_STATIC_V(size(scales)) == CUTE_STATIC_V(size(taccOcO_row)));
            #pragma unroll
            for (int mi = 0; mi < size(taccOcO_row); ++mi) {
                // 只有第一列的线程负责写入缩放因子，避免冗余写入
                if (get<1>(taccOcO_row(_0{})) == 0) {
                    sScale(get<0>(taccOcO_row(mi)), stage) = scales(mi);
                }
            }
        };

        // Lambda函数：等待pipeline数据就绪的消费者操作
        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        // === 序列长度和块处理初始化 ===
        int const seqlen_q = seqlen_info.seqlen_q;  // Query序列长度
        int const seqlen_k = seqlen_info.seqlen_k;  // Key序列长度
        int n_block = n_block_max - 1;              // 从最后一个块开始反向处理

        // === 创建注意力掩码对象 ===
        // 处理因果掩码、局部注意力窗口、序列长度掩码等
        flash::Mask<kBlockM, kBlockN, PackGQA, TiledMmaQK> mask(
            thread_idx, seqlen_q, seqlen_k, params.window_size_left, params.window_size_right, 0 /*sink_token_length*/,
            params.attention_chunk_divmod, params.qhead_per_khead_divmod, params.qhead_per_khead_mtp_divmod
        );

        // === Softcap（软限制）处理 ===
        float softcap_val = params.softcap_val;
        if constexpr (Has_softcap && Is_FP8) {
            // 对于FP8精度，需要考虑Q和K的去量化因子
            float const q_descale = params.ptr_q_descale == nullptr ? 1.0f : params.ptr_q_descale[bidb * get<0>(params.stride_q_descale) + bidh_kv * get<1>(params.stride_q_descale)];
            float const k_descale = params.ptr_k_descale == nullptr ? 1.0f : params.ptr_k_descale[bidb * get<0>(params.stride_k_descale) + bidh_kv * get<1>(params.stride_k_descale)];
            softcap_val *= q_descale * k_descale;
        }
        
        // Lambda函数：在掩码应用之前进行softcap处理
        // 注意：softcap必须在掩码之前应用，否则会将-inf转换为有限值，影响softmax计算
        auto scoremod_premask_fn = [&](auto& tSrS) {
            if constexpr (Has_softcap) { flash::apply_softcap(tSrS, softcap_val); }
        };

        // Lambda函数：将P矩阵写入共享内存
        auto write_P_to_smem = [&](auto& tOrP) {
            if constexpr (LargeHeadDimV) {
                // 同步barrier确保共享内存P区域可用
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PEmpty) /*id*/);
            }
            cute::copy(smem_tiled_copy_P, smem_thr_copy_P.retile_S(tOrP), tPsP);
        };

        // Lambda函数：在P写入完成后到达barrier
        auto arrive_on_P_write_barrier = [&] {
            cutlass::arch::fence_view_async_shared();  // 确保共享内存写入完成
            __syncwarp();  // 只需要warp级同步，因为每个warp使用自己的P值进行MmaPV
            if constexpr (LargeHeadDimV) {
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PFull) /*id*/);
            }
        };

        // === Q矩阵的barrier同步和旋转位置编码处理 ===
        auto &barrier_Q = shared_storage.pipelines.barrier_Q;
        if constexpr (!AppendKV) {
            // 标准模式：等待Q矩阵数据就绪
            barrier_Q.wait(work_idx % 2);
        } else {
            // AppendKV模式：可能需要应用旋转位置编码
            if (get<1>(params.shape_rotary) > 0) {  // 检查是否需要应用旋转位置编码
                // 定义旋转位置编码类型，根据是否为因果/局部注意力决定位置是否固定
                using Rotary_t = Rotary<kBlockM, kHeadDim, NumMmaThreadsQK, Element, !(Is_causal || Is_local) /*FixedPosition*/>;
                
                // 创建旋转位置编码对象
                Rotary_t rotary(params.ptr_rotary_cos, params.shape_rotary, params.stride_rotary_cos,
                                params.ptr_rotary_sin, params.stride_rotary_sin,
                                params.is_rotary_interleaved, thread_idx, seqlen_q,
                                seqlen_info.seqlen_rotary);
                                
                // 获取位置无关的Q张量视图
                Tensor sQ_pi = cute::as_position_independent_swizzle_tensor(sQ);
                int const qhead_per_khead = !PackGQA ? 1 : params.qhead_per_khead_divmod.divisor;
                
                if (params.is_rotary_interleaved) {
                    // 交错式旋转位置编码：cos和sin值交错存储
                    auto [tRrCos, tRrSin] = cute::conditional_return<!PackGQA>(
                        rotary.template load_cos_sin<true /*kInterleaved*/>(m_block),
                        rotary.template load_cos_sin_packgqa<true /*kInterleaved*/>(m_block, params.qhead_per_khead_divmod)
                    );
                    barrier_Q.wait(work_idx % 2);  // 等待Q数据就绪
                    rotary.apply_Q_interleaved(sQ_pi, tRrCos, tRrSin, m_block, qhead_per_khead);
                } else {
                    // 连续式旋转位置编码：cos和sin值分别连续存储
                    auto [tRrCosCont, tRrSinCont] = cute::conditional_return<!PackGQA>(
                        rotary.template load_cos_sin<false /*kInterleaved*/>(m_block),
                        rotary.template load_cos_sin_packgqa<false /*kInterleaved*/>(m_block, params.qhead_per_khead_divmod)
                    );
                    barrier_Q.wait(work_idx % 2);  // 等待Q数据就绪
                    rotary.apply_Q_contiguous(sQ_pi, tRrCosCont, tRrSinCont, m_block, qhead_per_khead);
                }
                
                // 共享内存fence，确保旋转后的Q对GMMA可见
                cutlass::arch::fence_view_async_shared();
                cutlass::arch::NamedBarrier::sync(NumMmaThreadsQK, static_cast<uint32_t>(FwdNamedBarriers::QueryRotated) /*id*/);
            } else {
                // 不需要旋转位置编码，直接等待Q数据就绪
                barrier_Q.wait(work_idx % 2);
            }
        }

        // === Q矩阵的寄存器存储模式处理 ===
        if constexpr (MmaQK_is_RS) {
            // 当Q*K使用寄存器存储模式时，需要将Q从共享内存拷贝到寄存器
            using SmemCopyAtomQ = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
            auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma_qk);
            auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
            Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
            Tensor tSsQ_copy_view = smem_thr_copy_Q.partition_S(cute::as_position_independent_swizzle_tensor(sQ));
            cute::copy(smem_tiled_copy_Q, tSsQ_copy_view, tSrQ_copy_view);
        }

        // === Flash Attention主计算循环 ===
        // 根据IntraWGOverlap标志选择不同的执行策略
        if constexpr (IntraWGOverlap) {
            // === Warp Group内重叠执行模式 ===
            // 这种模式允许在warp group内部重叠不同的计算阶段以提高效率
            
            // *** 第一次迭代的特殊处理 ***
            Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
            consumer_wait(pipeline_k, smem_pipe_read);  // 等待K矩阵数据就绪
            
            // 执行 Q * K^T 矩阵乘法，计算attention scores
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
            warpgroup_wait<0>();  // 等待GEMM完成
            pipeline_k.consumer_release(smem_pipe_read);  // 释放K矩阵pipeline
            
            // 如果启用了Q*V计算（某些特殊场景）
            if constexpr (HasQv) {
                shared_storage.pipelines.barrier_Qv.wait(work_idx % 2);
                consumer_wait(pipeline_v, smem_pipe_read);
                // 执行 Q * V 并累加到scores中
                flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma_qv, tSrQv, tSrV(_, _, _, smem_pipe_read.index()), tSrS);
            }
            
            // 应用softcap（如果启用）
            scoremod_premask_fn(tSrS);
            
            // 应用注意力掩码（因果掩码、局部掩码、序列长度掩码）
            mask.template apply<true /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS, m_block, n_block);

            // 计算softmax的最大值和缩放因子（第一次迭代）
            Tensor scores_scale = softmax.template max_get_scale</*Is_first=*/true, /*Check_inf=*/true>(tSrS);
            // 注意：第一次迭代时缩放因子为1.0，无需存储到共享内存

            // 执行在线softmax计算
            softmax.template online_softmax</*Is_first=*/true, /*Check_inf=*/true>(tSrS);
            
            // FP8精度处理：重新排列寄存器以匹配后续计算需求
            if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }
            
            // 将attention scores转换为适合P*V计算的格式
            Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
            Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
            convert_type_out(tOrP_acc, tOrP);  // 类型转换
            
            if constexpr (Is_FP8 && V_colmajor) { flash::permute_Aregs_fp8(tOrP); }
            if constexpr (!MmaPV_is_RS) { write_P_to_smem(tOrP); }      // 写入P矩阵到共享内存
            if constexpr (!MmaPV_is_RS) { arrive_on_P_write_barrier(); } // 到达写入完成barrier
            --n_block;  // 移动到下一个块

            // 初始化输出张量O（在RescaleOBeforeGemm模式下需要）
            clear(tOrO);
            // tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;

            // *** 前向计算步骤 Lambda 函数 ***
            // 每个步骤执行：当前n_block的QK计算，下一个n_block+1的PV计算，当前n_block的softmax
            auto fwd_step = [&](int const n_block, auto mask_fn, auto check_inf_type) {
                static constexpr bool Check_inf = decltype(check_inf_type)::value;
                
                // 保存V矩阵的pipeline状态，用于与K矩阵pipeline错峰处理
                PipelineState smem_pipe_read_v(smem_pipe_read.index(), smem_pipe_read.phase(), smem_pipe_read.count());
                ++smem_pipe_read;  // 推进pipeline状态
                
                // === 步骤1：Q * K^T 矩阵乘法 ===
                Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
                if (!UseSchedulerBarrier || warp_group_idx == 0) { consumer_wait(pipeline_k, smem_pipe_read); }
                warp_scheduler_barrier_sync();  // 同步调度器barrier
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
                
                // 如果启用了提前重缩放模式，需要在GEMM前重缩放输出O
                if constexpr (RescaleOBeforeGemm) { softmax.rescale_o(tOrO, scores_scale); }
                
                // === 步骤2：P * V 矩阵乘法（与QK计算重叠） ===
                if constexpr(!HasQv) {
                    if (!UseSchedulerBarrier || warp_group_idx == 0) { consumer_wait(pipeline_v, smem_pipe_read_v); }
                }
                // 执行 P * V 计算，累加到输出O中
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_pv, cute::conditional_return<MmaPV_is_RS>(tOrP, tOsP), tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
                
                warp_scheduler_barrier_arrive();  // 到达调度器barrier
                warpgroup_wait<1>();  // 等待第二个GEMM完成
                pipeline_k.consumer_release(smem_pipe_read);  // 释放K矩阵pipeline
                
                // === 处理Q*V计算（如果启用） ===
                if constexpr (HasQv) {
                    warpgroup_wait<0>();  // 等待第一个GEMM完成
                    pipeline_v.consumer_release(smem_pipe_read_v);  // 释放V矩阵pipeline
                    consumer_wait(pipeline_v, smem_pipe_read);
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma_qv, tSrQv, tSrV(_, _, _, smem_pipe_read.index()), tSrS);
                }
                
                // === 步骤3：Softmax处理 ===
                scoremod_premask_fn(tSrS);  // 应用softcap
                mask_fn(tSrS, n_block);     // 应用掩码
                
                // 计算新的最大值和缩放因子，更新在线softmax状态
                cute::copy(softmax.template max_get_scale</*Is_first=*/false, Check_inf>(tSrS), scores_scale);
                if constexpr (LargeHeadDimV) { store_scales(scores_scale, smem_pipe_read_v.index()); }
                
                // 执行在线softmax更新
                softmax.template online_softmax</*Is_first=*/false, Check_inf>(tSrS);
                
                // 释放V矩阵pipeline（如果未在前面释放）
                if constexpr (!HasQv) {
                    warpgroup_wait<0>();
                    pipeline_v.consumer_release(smem_pipe_read_v);  // release V
                }
                
                // === 步骤4：数据格式转换和存储 ===
                if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }
                convert_type_out(make_tensor(tSrS.data(), tOrP.layout()), tOrP);  // 转换数据类型
                if constexpr (Is_FP8 && V_colmajor) { flash::permute_Aregs_fp8(tOrP); }
                
                if constexpr (!MmaPV_is_RS) { write_P_to_smem(tOrP); }  // 写入P到共享内存
                
                // 重缩放输出O（根据不同的缩放策略）
                if constexpr (!RescaleOBeforeGemm) { softmax.rescale_o(tOrO, scores_scale); }
                
                if constexpr (!MmaPV_is_RS) { arrive_on_P_write_barrier(); }  // 到达写入完成barrier
            };

            // === 根据掩码类型分别处理不同的块范围 ===
            
            // *** 处理需要因果掩码或局部掩码的块 ***
            if constexpr (Is_causal || Is_local) {
                auto mask_fn = [&](auto& tSrS, int n_block) { 
                    mask.template apply<false /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS, m_block, n_block); 
                };
                // 计算需要应用因果/局部掩码的最小块索引
                int const n_block_min_causal_local_mask = BlockMN_t::get_n_block_min_causal_local_mask(
                    seqlen_info, m_block, n_block_min, params.window_size_right,
                    params.attention_chunk_divmod, params.qhead_per_khead_divmod);
                #pragma unroll 1
                for (; n_block >= n_block_min_causal_local_mask; --n_block) {
                    fwd_step(n_block, mask_fn, cute::true_type{} /*check_inf*/);  // 需要检查无穷值
                }
            }

            // *** 处理不需要掩码的块（性能优化）***
            int const n_block_min_before_local_mask = BlockMN_t::get_n_block_min_before_local_mask(
                seqlen_info, m_block, n_block_min, params.window_size_left,
                params.attention_chunk_divmod, params.qhead_per_khead_divmod);
            auto no_mask_fn = [](auto& tSrS, int n_block) { };  // 空函数，无掩码操作
            #pragma unroll 1
            for (; n_block >= n_block_min_before_local_mask; --n_block) {
                fwd_step(n_block, no_mask_fn, cute::false_type{} /*check_inf*/);  // 无需检查无穷值
            }
            
            // *** 处理局部注意力左侧的掩码块 ***
            if constexpr (Is_local) {
                auto local_mask_fn = [&](auto& tSrS, int n_block) { 
                    mask.template apply<false /*Seqlenk_mask*/, false /*Causal_mask*/, Is_local>(tSrS, m_block, n_block); 
                };
                #pragma unroll 1
                for (; n_block >= n_block_min; --n_block) {
                    fwd_step(n_block, local_mask_fn, cute::bool_constant<Is_local>{} /*check_inf*/);
                }
            }
            
            // === 最终处理和清理阶段 ===
            
            // 通知生产者共享内存Q区域已就绪
            cutlass::arch::NamedBarrier::arrive(NumMmaThreadsQK + (Use_TMA_Q ? cutlass::NumThreadsPerWarp : NumProducerThreads), static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);
            
            // 根据缩放策略重缩放输出O
            if constexpr (RescaleOBeforeGemm) { softmax.rescale_o(tOrO, scores_scale); }
            
            // *** 执行最后一次P*V计算 ***
            if constexpr (!HasQv) { consumer_wait(pipeline_v, smem_pipe_read); }
            flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_pv, cute::conditional_return<MmaPV_is_RS>(tOrP, tOsP), tOrV(_, _, _, smem_pipe_read.index()), tOrO);
            
            // 获取V矩阵的去量化因子（FP8精度）
            float const v_descale = !Is_FP8 || params.ptr_v_descale == nullptr ? 1.0f : params.ptr_v_descale[bidb * get<0>(params.stride_v_descale) + bidh_kv * get<1>(params.stride_v_descale)];
            
            // 完成softmax计算并获取最终缩放因子
            cute::copy(softmax.finalize(v_descale), scores_scale);
            
            // 处理大头维度V的缩放因子存储
            if constexpr (LargeHeadDimV) {
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PEmpty) /*id*/);
                store_scales(scores_scale, smem_pipe_read.index());
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PFull) /*id*/);
            }
            
            // 等待最后的GEMM完成并释放V矩阵pipeline
            warpgroup_wait<0>();
            pipeline_v.consumer_release(smem_pipe_read);  // 释放V，否则生产者会挂起
            
            // 应用最终的缩放因子到输出O
            softmax.rescale_o(tOrO, scores_scale);
            
            // FP8精度的输出重排列
            if constexpr (Is_FP8 && !V_colmajor) { flash::permute_output_fp8(tOrO); }
            
            ++smem_pipe_read;  // 推进pipeline状态

        } else {  
            // === 非Warp Group重叠模式 ===
            // 这种模式使用更简单的顺序执行策略，适用于某些硬件配置

            warp_scheduler_barrier_sync();  // 同步调度器barrier，确保所有warp处于相同状态

            // === 定义前向传播单步函数fwd_step ===
            // 这是flash attention计算的核心函数，实现了Q*K^T -> Softmax -> *V的完整流程
            auto fwd_step = [&](int const n_block, auto mask_fn, auto is_first_iter_type, auto check_inf_type) {
                // 编译时常量：是否为第一次迭代和是否检查无穷大
                static constexpr bool Is_first_iter = decltype(is_first_iter_type)::value;
                static constexpr bool Check_inf = decltype(check_inf_type)::value;
                
                // 保存当前pipeline读取位置，用于后续的scale存储
                auto smem_pipe_read_prev = smem_pipe_read;
                // 如果不是第一次迭代，推进pipeline读取位置
                if constexpr (!Is_first_iter) { ++smem_pipe_read; }
                // === 第一阶段：Q*K^T矩阵乘法 ===
                // 创建用于存储Q*K^T结果的张量S（注意力分数）(64, 64) Score
                Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
                // 等待K矩阵数据在pipeline中准备就绪
                consumer_wait(pipeline_k, smem_pipe_read);
                // 执行Q*K^T矩阵乘法，zero_init=true表示结果初始化为0
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
                
                // === 处理不同的执行路径：有无Qv重叠 ===
                if constexpr (!HasQv) {
                    // 没有Q和V重叠的情况：简单的顺序执行
                    warp_scheduler_barrier_arrive();  // 发出barrier信号
                    warpgroup_wait<0>();              // 等待warpgroup 0完成
                    pipeline_k.consumer_release(smem_pipe_read);  // 释放K数据的pipeline资源
                } else {
                    // 有Q和V重叠的情况：需要更复杂的同步
                    if constexpr (Is_first_iter) {
                        // 第一次迭代时等待Qv barrier
                        shared_storage.pipelines.barrier_Qv.wait(work_idx % 2);
                    }
                    // 等待V矩阵数据准备就绪
                    consumer_wait(pipeline_v, smem_pipe_read);
                    // 执行第二个GEMM：Qv*V，zero_init=false表示累加到现有结果
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_qv, tSrQv, tSrV(_, _, _, smem_pipe_read.index()), tSrS);
                    warp_scheduler_barrier_arrive();  // 发出barrier信号
                    warpgroup_wait<1>();              // 等待warpgroup 1完成
                    pipeline_k.consumer_release(smem_pipe_read);  // 释放K数据
                    warpgroup_wait<0>();              // 等待warpgroup 0完成
                }
                // === 第二阶段：注意力分数处理和Softmax ===
                // 应用预掩码函数（如RoPE等位置编码）
                scoremod_premask_fn(tSrS);
                // 应用mask函数（causal mask、local mask等）
                mask_fn(tSrS, n_block);
                // 计算softmax的max和scale，用于数值稳定的online softmax
                Tensor scores_scale = softmax.template max_get_scale</*Is_first=*/Is_first_iter, Check_inf>(tSrS);
                // 对于大head维度且非第一次迭代，存储scale值用于后续rescale
                if constexpr (LargeHeadDimV && !Is_first_iter) { store_scales(scores_scale, smem_pipe_read_prev.index()); }
                // 执行online softmax算法，将注意力分数转换为概率
                softmax.template online_softmax</*Is_first=*/Is_first_iter, Check_inf>(tSrS);
                
                // === 第三阶段：准备P*V矩阵乘法 ===
                // 如果使用FP8且V不是列主序，需要重排寄存器布局
                if constexpr (Is_FP8 && !V_colmajor) { flash::permute_Cregs_fp8(tSrS); }
                // 创建用于P*V运算的累加器张量
                Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
                // 创建输出类型的P张量
                Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
                // 将累加器类型转换为输出类型
                convert_type_out(tOrP_acc, tOrP);
                // 如果使用FP8且V是列主序，重排A寄存器布局
                if constexpr (Is_FP8 && V_colmajor) { flash::permute_Aregs_fp8(tOrP); }
                // === 第四阶段：内存管理和同步 ===
                // 如果不使用寄存器spilling，将P写入共享内存
                if constexpr (!MmaPV_is_RS) { write_P_to_smem(tOrP); }
                // 如果不是第一次迭代，用新的scale重新缩放输出O
                if constexpr (!Is_first_iter) { softmax.rescale_o(tOrO, scores_scale); }
                // 根据不同的配置发出P写入barrier信号
                if constexpr (!MmaPV_is_RS && !MmaPV_use_RS_WG1) { arrive_on_P_write_barrier(); }
                // 如果没有Qv重叠，等待V数据准备
                if constexpr (!HasQv) { consumer_wait(pipeline_v, smem_pipe_read); }
                
                // === 第五阶段：P*V矩阵乘法 ===
                warp_scheduler_barrier_sync();  // 同步所有warp
                if constexpr (!MmaPV_use_RS_WG1) {
                    // 标准P*V矩阵乘法：P*V -> O
                    // zero_init根据是否第一次迭代决定：第一次清零，后续累加
                    flash::gemm</*zero_init=*/Is_first_iter, /*wg_wait=*/-1>(tiled_mma_pv, cute::conditional_return<MmaPV_is_RS>(tOrP, tOsP), tOrV(_, _, _, smem_pipe_read.index()), tOrO);
                } else {
                    // 使用寄存器spilling的warpgroup 1模式
                    TiledMmaPV_RS tiled_mma_pv_rs;
                    flash::gemm</*zero_init=*/Is_first_iter, /*wg_wait=*/-1>(tiled_mma_pv_rs, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
                }
                
                // === 第六阶段：清理和同步 ===
                // 根据配置发出P写入barrier信号
                if constexpr (!MmaPV_is_RS && MmaPV_use_RS_WG1) { arrive_on_P_write_barrier(); }
                warpgroup_wait<0>();  // 等待warpgroup 0完成
                pipeline_v.consumer_release(smem_pipe_read);  // 释放V数据的pipeline资源
            };

            // === 执行不同mask策略的迭代 ===
            
            // 第一次迭代：需要应用序列长度mask和其他所有mask
            auto first_iter_mask_fn = [&](auto& tSrS, int n_block) { 
                mask.template apply<true /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS, m_block, n_block); 
            };
            fwd_step(n_block, first_iter_mask_fn, cute::true_type{} /*is_first_iter*/, cute::true_type{} /*check_inf*/);
            --n_block;  // 移动到下一个block
            
            // 处理因果mask或局部mask的迭代
            if constexpr (Is_causal || Is_local) {
                // 定义mask函数：不需要序列长度mask，但需要因果/局部mask
                auto mask_fn = [&](auto& tSrS, int n_block) { 
                    mask.template apply<false /*Seqlenk_mask*/, Is_causal, Is_local>(tSrS, m_block, n_block); 
                };
                // 计算需要应用因果/局部mask的最小block编号
                int const n_block_min_causal_local_mask = BlockMN_t::get_n_block_min_causal_local_mask(
                    seqlen_info, m_block, n_block_min, params.window_size_right,
                    params.attention_chunk_divmod, params.qhead_per_khead_divmod);
                // 循环处理需要因果/局部mask的blocks
                #pragma unroll 1
                for (; n_block >= n_block_min_causal_local_mask; --n_block) {
                    fwd_step(n_block, mask_fn, cute::false_type{} /*is_first_iter*/, cute::true_type{} /*check_inf*/);
                }
            }
            // 计算局部mask左边界之前的最小block编号
            int const n_block_min_before_local_mask = BlockMN_t::get_n_block_min_before_local_mask(
                seqlen_info, m_block, n_block_min, params.window_size_left,
                params.attention_chunk_divmod, params.qhead_per_khead_divmod);
            
            // 处理不需要任何mask的迭代（完全可见的blocks）
            auto no_mask_fn = [](auto& tSrS, int n_block) { /* 空函数，不应用任何mask */ };
            #pragma unroll 1
            for (; n_block >= n_block_min_before_local_mask; --n_block) {
                // 不是第一次迭代，不检查无穷大（因为没有mask，数值应该稳定）
                fwd_step(n_block, no_mask_fn, cute::false_type{} /*is_first_iter*/, cute::false_type{} /*check_inf*/);
            }
            
            // 处理局部attention左侧的mask迭代
            if constexpr (Is_local) {
                // 定义局部mask函数：只应用局部mask，不应用因果mask
                auto local_mask_fn = [&](auto& tSrS, int n_block) { 
                    mask.template apply<false /*Seqlenk_mask*/, false /*Causal_mask*/, Is_local>(tSrS, m_block, n_block); 
                };
                #pragma unroll 1
                for (; n_block >= n_block_min; --n_block) {
                    // 对于局部mask，check_inf取决于是否为局部attention
                    fwd_step(n_block, local_mask_fn, cute::false_type{} /*is_first_iter*/, cute::bool_constant<Is_local>{} /*check_inf*/);
                }
            }
            // === 最终处理和清理 ===
            warp_scheduler_barrier_arrive();  // 发出调度器barrier信号
            
            // 通知生产者共享内存中的Q数据已经可以被释放
            cutlass::arch::NamedBarrier::arrive(NumMmaThreadsQK + (Use_TMA_Q ? cutlass::NumThreadsPerWarp : NumProducerThreads), 
                                              static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);
            
            // 获取V的descale因子（用于FP8量化）
            float const v_descale = !Is_FP8 || params.ptr_v_descale == nullptr ? 1.0f : 
                params.ptr_v_descale[bidb * get<0>(params.stride_v_descale) + bidh_kv * get<1>(params.stride_v_descale)];
            
            // 完成softmax计算，获得最终的scale
            Tensor scores_scale = softmax.finalize(v_descale);
            
            // 对于大head维度，需要额外的同步和scale存储
            if constexpr (LargeHeadDimV) {
                // 等待P数据清空
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PEmpty) /*id*/);
                // 存储最终的scale值
                store_scales(scores_scale, smem_pipe_read.index());
                // 通知P数据已满
                cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PFull) /*id*/);
            }
            
            // 用最终scale重新缩放输出O
            softmax.rescale_o(tOrO, scores_scale);
            
            // 如果使用FP8且V不是列主序，需要重排输出
            if constexpr (Is_FP8 && !V_colmajor) { flash::permute_output_fp8(tOrO); }
            
            ++smem_pipe_read;  // 推进pipeline读取状态，为下一轮做准备
        }
        
        // === 函数结尾处理 ===
        ++work_idx;  // 增加工作索引，为下一次调用做准备
        return true; // 返回true表示成功完成计算
    }  // mma函数结束

    /**
     * mma_pv - 执行注意力权重矩阵P和值矩阵V的矩阵乘法运算 (P@V)
     * 
     * 这是Flash Attention算法中的关键函数，专门处理较大头维度的情况。
     * 在这种情况下，计算被分割到不同的warp group中：
     * - warp group 1: 执行 Q@K^T 和 Softmax 计算，生成注意力权重P
     * - warp group 2: 执行 P@V 乘法，计算最终的注意力输出O
     * 
     * @tparam SharedStorage 共享内存存储类型
     * @tparam FrgTensorO 输出张量fragment类型，必须驻留在寄存器内存中
     * @tparam Softmax softmax操作类型
     * 
     * @param params 主循环参数，包含各种配置信息
     * @param pipeline_v V矩阵的流水线，用于异步数据传输
     * @param smem_pipe_read 共享内存流水线读取状态
     * @param tOrO 输出张量O的fragment，存储最终的注意力结果
     * @param softmax softmax操作对象，用于处理注意力权重的缩放
     * @param thread_idx 当前线程在warp group中的索引
     * @param seqlen_info 序列长度信息，用于处理变长序列和掩码
     * @param block_coord 当前处理的块坐标 (m_block, n_block, batch_id, split_idx)
     * @param shared_storage 共享内存存储区域
     * 
     * @return bool 返回true表示成功完成计算，false表示由于掩码等原因跳过计算
     */
    template <typename SharedStorage, typename FrgTensorO, typename Softmax>
    CUTLASS_DEVICE bool
    mma_pv(Params const& params,
           MainloopPipelineV pipeline_v,
           PipelineState& smem_pipe_read,
           FrgTensorO& tOrO,
           Softmax& softmax,
           int const thread_idx,
           SeqlenInfo_t const& seqlen_info,
           cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
           SharedStorage& shared_storage
           ) {
        // 确保输出张量O驻留在寄存器内存中以获得最佳性能
        static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");
        
        // === 解析块坐标信息 ===
        // 不能使用结构化绑定，因为lambda表达式无法捕获结构化绑定的变量
        int const m_block = get<0>(block_coord);      // 行块索引（query block）
        int const bidb = get<2>(block_coord);         // batch索引
        int const split_idx = get<3>(block_coord);    // 分割索引（用于split-k优化）
        
        // === 计算有效的列块范围 ===
        // 根据因果掩码、局部注意力窗口、变长序列等约束确定需要处理的key块范围
        auto [n_block_min, n_block_max] = BlockMN_t::get_n_block_min_max(
            seqlen_info, m_block, bidb, split_idx, params.num_splits,
            params.window_size_left, params.window_size_right, params.attention_chunk_divmod,
            params.qhead_per_khead_divmod);
        
        // === 提前终止检查 ===
        // 如果没有有效的key块需要处理，直接返回
        if constexpr (Is_causal || Is_local || Varlen || Split) {
            if (n_block_max <= n_block_min) { return false; }
        }

        // === 设置共享内存张量视图 ===
        // 创建指向共享内存中V矩阵、P矩阵和缩放因子的张量视图
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVtMma{});  // V矩阵（转置后用于MMA）
        Tensor sP = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutP{});      // P矩阵（softmax输出的注意力权重）
        Tensor sScale = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_scale.data()), SmemLayoutScale{}); // 缩放因子
        
        // === 设置warp group级别的MMA布局 ===
        // 计算warp group数量并创建线程布局
        static constexpr int MmaWarpGroups = size(TiledMmaPV{}) / cutlass::NumThreadsPerWarpGroup;
        Layout warp_group_thread_layout = make_layout(make_shape(Int<MmaWarpGroups>{}),
                                                      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

        // 获取当前线程所属的warp group索引
        int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
        
        // === 创建P@V矩阵乘法的tiled MMA对象 ===
        TiledMmaPV tiled_mma_pv;  // P*V矩阵乘法的MMA配置
        auto wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx));

        // === 分配矩阵fragments ===
        // 为当前warp group分配V矩阵和P矩阵的fragment
        Tensor tOrV = wg_mma_pv.partition_fragment_B(sV);   // V矩阵fragment（作为乘法的右操作数B）
        Tensor tOsP = wg_mma_pv.partition_fragment_A(sP);   // P矩阵fragment（作为乘法的左操作数A）

        // === 设置输出累加器和缩放因子加载 ===
        // 为缩放因子加载创建线程级别的MMA slice（假设thread_idx为128取模以适应warp group）
        auto thread_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx % cutlass::NumThreadsPerWarpGroup);
        Tensor taccOcO = thread_mma_pv.partition_C(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));
        
        // 转换累加器布局为行列格式以便于缩放操作
        Tensor taccOcO_rowcol = make_tensor(taccOcO.data(), flash::convert_layout_acc_rowcol(taccOcO.layout()));
        Tensor taccOcO_row = taccOcO_rowcol(_, _0{});  // 提取行维度用于缩放
        
        // === Lambda函数：从共享内存加载缩放因子 ===
        auto load_scales = [&](auto& scales, int stage) {
            static_assert(CUTE_STATIC_V(size(scales)) == CUTE_STATIC_V(size(taccOcO_row)));
            #pragma unroll
            for (int mi = 0; mi < size(taccOcO_row); ++mi) {
                scales(mi) = sScale(get<0>(taccOcO_row(mi)), stage);  // 从对应stage加载缩放因子
            }
        };

        // === 初始化输出张量（已在外部清零） ===
        // clear(tOrO);  // 输出张量在外部已经初始化
        // tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;  // 设置初始累加模式

        typename Softmax::TensorT scores_scale;  // 存储softmax缩放因子的张量

        // === 主计算循环：从最后一个key块开始向前处理 ===
        int n_block = n_block_max - 1;  // 从最后一个有效key块开始
        
        // === 处理第一个key块 ===
        // 如果不使用QV流水线，等待V矩阵数据准备就绪
        if constexpr (!HasQv) { pipeline_v.consumer_wait(smem_pipe_read); }
        
        // 同步等待P矩阵数据准备完成
        cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PFull) /*id*/);
        
        // 执行第一次P@V矩阵乘法（zero_init=true表示清零累加器）
        flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma_pv, tOsP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
        
        // 通知P矩阵缓冲区可以重用
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PEmpty) /*id*/);
        pipeline_v.consumer_release(smem_pipe_read);  // 释放V矩阵缓冲区
        --n_block;  // 移到下一个key块

        // === 主循环：处理剩余的key块 ===
        #pragma unroll 1  // 限制循环展开以控制寄存器使用
        for (; n_block >= n_block_min; --n_block) {
            // 等待下一个P矩阵准备完成
            cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PFull) /*id*/);
            
            // 加载当前stage的缩放因子
            load_scales(scores_scale, smem_pipe_read.index());
            
            // 使用缩放因子重新缩放之前累积的输出O
            // 这是Flash Attention算法的关键步骤：在线更新注意力输出
            softmax.rescale_o(tOrO, scores_scale);
            
            // 前进到下一个流水线stage
            ++smem_pipe_read;
            
            // 如果不使用QV流水线，异步等待下一个V矩阵数据
            if constexpr (!HasQv) {
                auto barrier_token = pipeline_v.consumer_try_wait(smem_pipe_read);
                pipeline_v.consumer_wait(smem_pipe_read, barrier_token);
            }
            
            // 执行P@V矩阵乘法并累加到输出O中（zero_init=false表示累加模式）
            flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma_pv, tOsP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
            
            // 通知P矩阵缓冲区可以重用
            cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PEmpty) /*id*/);
            pipeline_v.consumer_release(smem_pipe_read);  // 释放V矩阵缓冲区
        };
        
        // === 最终处理：应用最后的缩放 ===
        cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PFull) /*id*/);
        load_scales(scores_scale, smem_pipe_read.index());  // 加载最后的缩放因子
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<uint32_t>(FwdNamedBarriers::PEmpty) /*id*/);
        softmax.rescale_o(tOrO, scores_scale);  // 应用最终缩放
        
        // === FP8特殊处理 ===
        // 如果使用FP8且V矩阵不是列主序，需要重新排列输出
        if constexpr (Is_FP8 && !V_colmajor) { flash::permute_output_fp8(tOrO); }
        
        ++smem_pipe_read;  // 前进流水线状态
        return true;       // 返回成功
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE bool
    load_kv_new(Params const& params,
         MainloopPipelineKVNew pipeline_k_new,
         MainloopPipelineKVNew pipeline_v_new,
         PipelineState& smem_pipe_write,
         SharedStorage &shared_storage,
         SeqlenInfo_t const& seqlen_info,
         cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
         int const work_idx
         ) {

        auto [m_block, bidh, bidb, split_idx] = block_coord;
        auto [n_block_new_min, n_block_new_max] = BlockMN_t::get_n_block_k_new_min_max(
            seqlen_info, m_block, bidb, split_idx, params.num_splits,
            params.window_size_left, params.window_size_right, params.attention_chunk_divmod,
            params.qhead_per_khead_divmod);

        if (n_block_new_max <= n_block_new_min) { return false; }

        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sVt = [&] {
            if constexpr (!Transpose_V) {
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVt{});
            } else {
                return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_vt.data()), SmemLayoutVt{});
            }
        }();

        // int const thread_idx = threadIdx.x % NumProducerThreads;
        int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;

        // Prepare the TMA loads
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

        bool const is_varlen_k_new = Varlen && params.cu_seqlens_k_new;
        Tensor mKnew_TMA = params.tma_load_K_new.get_tma_tensor(params.shape_K_new)(_, _, bidh_kv, !is_varlen_k_new ? bidb : 0);
        auto shape_Vnew = make_shape(params.headdim_v, get<0>(params.shape_K_new), get<2>(params.shape_K_new), get<3>(params.shape_K_new));
        Tensor mVnewt_TMA = params.tma_load_V_new.get_tma_tensor(shape_Vnew)(_, _, bidh_kv, !is_varlen_k_new ? bidb : 0);

        Tensor gKnew_TMA = local_tile(domain_offset(make_coord(seqlen_info.offset_k_new, _0{}), mKnew_TMA), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
        Tensor gVnewt_TMA = local_tile(domain_offset(make_coord(_0{}, seqlen_info.offset_k_new), mVnewt_TMA), select<1, 2>(TileShape_MNK_PV{}), make_coord(_0{}, _));  // (K, N, _)

        auto block_tma_K_new = params.tma_load_K_new.get_slice(cluster_local_block_id.x);
        Tensor tKgKnew_TMA = group_modes<0, 3>(block_tma_K_new.partition_S(gKnew_TMA));  // (TMA, k)
        Tensor tKsK_TMA = group_modes<0, 3>(block_tma_K_new.partition_D(sK));  // (TMA, PIPE)
        auto block_tma_V_new = params.tma_load_V_new.get_slice(cluster_local_block_id.x);
        Tensor tVgVnewt_TMA = group_modes<0, 3>(block_tma_V_new.partition_S(gVnewt_TMA));  // (TMA, k)
        Tensor tVsVt_TMA = group_modes<0, 3>(block_tma_V_new.partition_D(sVt));  // (TMA, PIPE)

        uint16_t mcast_mask_kv = 0;
        if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
            auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
            for (int m = 0; m < size<0>(block_layout); ++m) {
                mcast_mask_kv |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
            }
        }

        auto load_K_new = [&] (int const n_block, auto const& smem_pipe_write) {
            pipeline_k_new.producer_acquire(smem_pipe_write);
            copy(params.tma_load_K_new.with(*pipeline_k_new.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_FIRST),
                tKgKnew_TMA(_, n_block), tKsK_TMA(_, smem_pipe_write.index()));
        };

        auto load_V_new = [&] (int const n_block, auto const& smem_pipe_write) {
            pipeline_v_new.producer_acquire(smem_pipe_write);
            copy(params.tma_load_V_new.with(*pipeline_v_new.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_FIRST),
                tVgVnewt_TMA(_, n_block), tVsVt_TMA(_, smem_pipe_write.index()));
        };

        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        // If this is true, we're guaranteed that only the first warp will execute this function
        static constexpr bool SingleProducerWarp = NumProducerThreads == cutlass::NumThreadsPerWarp;
        bool should_load_KV = (SingleProducerWarp || warp_idx_in_warpgroup == 0) && cute::elect_one_sync();

        int n_block = n_block_new_max - 1;
        // Need to wait for barrier_O even before load_K_new since the pipelines for AppendKV
        // and the main attention are not the same. We want to make sure the consumers
        // have finished reading all smem_k and smem_v for the previous iteration.
        shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);
        if (should_load_KV) { load_K_new(n_block, smem_pipe_write); }
        // if (thread_idx == 0) { printf("Producer: Done loading K, n_block = %d, n_block_new_min = %d\n", n_block, n_block_new_min); }
        if (should_load_KV) { load_V_new(n_block, smem_pipe_write); }
        // if (thread_idx == 0) { printf("Producer: Done loading V, n_block = %d, n_block_new_min = %d\n", n_block, n_block_new_min); }
        ++smem_pipe_write;
        --n_block;
        // if (thread_idx == 0) { printf("Producer: before for loop\n"); }
        #pragma unroll 1
        for (; n_block >= n_block_new_min; --n_block) {
            if (should_load_KV) {
                load_K_new(n_block, smem_pipe_write);
                // if (thread_idx == 0) { printf("Producer: Done loading K, n_block = %d, n_block_new_min = %d\n", n_block, n_block_new_min); }
                load_V_new(n_block, smem_pipe_write);
                // if (thread_idx == 0) { printf("Producer: Done loading V, n_block = %d, n_block_new_min = %d\n", n_block, n_block_new_min); }
            }
            ++smem_pipe_write;
        }
        // if (thread_idx == 0) { printf("Producer: after for loop\n"); }
        // At the end, all threads have the correct smem_pipe_write.
        return true;
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE bool
    store_kv_new(Params const& params,
                 MainloopPipelineKVNew pipeline_k_new,
                 MainloopPipelineKVNew pipeline_v_new,
                 PipelineState& smem_pipe_read,
                 int const thread_idx,
                 SharedStorage &shared_storage,
                 SeqlenInfo_t const& seqlen_info,
                 cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord
    ) {
        auto [m_block, bidh, bidb, split_idx] = block_coord;
        auto [n_block_new_min, n_block_new_max] = BlockMN_t::get_n_block_k_new_min_max(
            seqlen_info, m_block, bidb, split_idx, params.num_splits,
            params.window_size_left, params.window_size_right, params.attention_chunk_divmod,
            params.qhead_per_khead_divmod);
        if (n_block_new_max <= n_block_new_min) { return false; }

        // as_position_independent_swizzle_tensor makes address calculation easier
        Tensor sK = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{}));
        // We want to use SmemLayoutVCpAsync to have shape (kBlockN, kHeadDim) instead of (kHeadDim, kBlockN)
        Tensor sV = [&] {
            if constexpr (!Transpose_V) {
                return cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVCpAsync{}));
            } else {
                return cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_vt.data()), SmemLayoutVCpAsync{}));
            }
        }();

        int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;
        int const bidb_kv = params.kv_batch_idx == nullptr ? bidb : params.kv_batch_idx[bidb];

        bool const is_varlen_k = Varlen && params.cu_seqlens_k;
        Tensor mK = make_tensor(make_gmem_ptr(params.ptr_K), params.shape_K, params.stride_K)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);
        auto shape_V = make_shape(params.headdim_v, get<0>(params.shape_K), get<2>(params.shape_K), get<3>(params.shape_K));
        Tensor mV = make_tensor(make_gmem_ptr(params.ptr_V), shape_V, params.stride_V)(_, _, bidh_kv, !is_varlen_k ? bidb_kv : 0);

        int const offset_k = seqlen_info.offset_k + seqlen_info.seqlen_k_og;
        Tensor gK = local_tile(domain_offset(make_coord(offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
        Tensor gV = local_tile(domain_offset(make_coord(offset_k, _0{}), mV), select<2, 1>(TileShape_MNK_PV{}), make_coord(_, _0{}));  // (N, K_v, _)

        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        static constexpr int kHeadDim = get<2>(TileShape_MNK{});
        int const seqlen_k_new = seqlen_info.seqlen_k_new;
        using Rotary_t = Rotary<kBlockN, kHeadDim, NumMmaThreads, Element>;
        Rotary_t rotary(params.ptr_rotary_cos, params.shape_rotary, params.stride_rotary_cos,
                        params.ptr_rotary_sin, params.stride_rotary_sin,
                        params.is_rotary_interleaved, thread_idx, seqlen_k_new,
                        seqlen_info.seqlen_rotary);

        // This is used to index into the batch dimension of mK and mV
        int const bidb_kv_idx = !is_varlen_k && !params.ptr_pagetable ? bidb_kv : 0;

        using PagedKVManager_t = PagedKVManager<get<1>(TileShape_MNK{}), get<2>(TileShape_MNK{}), get<1>(TileShape_MNK_PV{}), NumMmaThreads, Element, true /*KV_Same_Iter*/, 2 /*LoadsPerRow_LB*/>;
        PagedKVManager_t paged_kv_manager(
            params.ptr_pagetable, params.shape_pagetable, params.stride_pagetable,
            params.ptr_K, params.shape_K, params.stride_K,
            params.ptr_V, params.headdim_v, params.stride_V,
            params.page_size_divmod, params.blockN_per_page_size_divmod,
            bidb_kv, bidh_kv, thread_idx, seqlen_k_new, offset_k, bidb_kv_idx
            // passing offset_k instead of leftpad_k will move the PageTable pointer to the right position
        );

        if constexpr (UseSchedulerBarrier) {
            // WG1 already got the very first signal from mma_init(), but we'll be using the same NamedBarrier.
            // So we'll need to "cancel it out" here and then re-signal it at the end.
            if (flash::canonical_warp_group_idx_nosync() == 1) {
                cutlass::arch::NamedBarrier::sync(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) /*id*/);
            }
        }

        static_assert(std::is_same_v<GmemLayoutAtom, typename Rotary_t::LayoutAtom>);
        static_assert(!PagedKVNonTMA || std::is_same_v<GmemLayoutAtom, typename PagedKVManager_t::GmemLayoutAtomKVCpAsync>);
        GmemTiledCopyAppendKV gmem_tiled_copy_kv;
        auto gmem_thr_copy_kv = gmem_tiled_copy_kv.get_thread_slice(thread_idx);
        Tensor tKsK = gmem_thr_copy_kv.partition_S(sK);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tKgK = gmem_thr_copy_kv.partition_D(gK);
        Tensor tVsV = gmem_thr_copy_kv.partition_S(sV);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tVgV = gmem_thr_copy_kv.partition_D(gV);
        Tensor cK = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tKcK = gmem_thr_copy_kv.partition_D(cK);
        Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK)));
        #pragma unroll
        for (int k = 0; k < size(tKpK); ++k) { tKpK(k) = get<1>(tKcK(_0{}, _0{}, k)) < get<1>(params.shape_K); }
        Tensor cV = cute::make_identity_tensor(select<2, 1>(TileShape_MNK_PV{}));  // (BLK_N,BLK_K_V) -> (blk_n,blk_k_v)
        Tensor tVcV = cute::conditional_return<SameHeadDim>(tKcK, gmem_thr_copy_kv.partition_D(cV));
        Tensor tVpV_ = make_tensor<bool>(make_shape(size<2>(tVsV)));
        #pragma unroll
        for (int k = 0; k < size(tVpV_); ++k) { tVpV_(k) = get<1>(tVcV(_0{}, _0{}, k)) < params.headdim_v; }
        Tensor tVpV = cute::conditional_return<SameHeadDim>(tKpK, tVpV_);

        auto store_K = [&] (int const n_block, auto const& smem_pipe_read) {
            int const n_limit = std::min(seqlen_k_new - n_block * kBlockN, kBlockN);
            if (get<1>(params.shape_rotary) <= 0) {
                pipeline_k_new.consumer_wait(smem_pipe_read);
                Tensor tKsK_cur = tKsK(_, _, _, smem_pipe_read.index());
                if constexpr (!PagedKVNonTMA) {
                    Tensor tKgK_cur = tKgK(_, _, _, n_block);
                    // Clear_OOB_K must be false since we don't want to write zeros to gmem
                    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                        gmem_tiled_copy_kv, tKsK_cur, tKgK_cur, tKcK, tKpK, std::min(seqlen_k_new - n_block * kBlockN, kBlockN)
                    );
                } else {
                    paged_kv_manager.store_K(n_block, tKsK_cur);
                }
            } else {
                Tensor gK_cur = gK(_, _, n_block);
                auto tPrKPtr = cute::conditional_return<PagedKVNonTMA>(paged_kv_manager.compute_K_ptr(), nullptr);
                if (params.is_rotary_interleaved) {
                    auto [tRrCos, tRrSin] = rotary.template load_cos_sin<true /*kInterleaved*/>(n_block);
                    pipeline_k_new.consumer_wait(smem_pipe_read);
                    rotary.template apply_K_interleaved<PagedKVNonTMA>(sK(_, _, smem_pipe_read.index()), gK_cur, tKpK, tRrCos, tRrSin, tPrKPtr, n_block);
                } else {
                    auto [tRrCosCont, tRrSinCont] = rotary.template load_cos_sin<false /*kInterleaved*/>(n_block);
                    pipeline_k_new.consumer_wait(smem_pipe_read);
                    rotary.template apply_K_contiguous<PagedKVNonTMA>(sK(_, _, smem_pipe_read.index()), gK_cur, tKpK, tRrCosCont, tRrSinCont, tPrKPtr, n_block, get<1>(params.shape_K));
                }
            }
            // Without this fence I'm getting race condition when seqlen_k is large
            cutlass::arch::fence_view_async_shared();
            // Very important: PipelineTmaAsync::consumer_release assumes that the warpgroup is synchronized
            // before calling.
            cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + flash::canonical_warp_group_idx_nosync() /*id*/);
            pipeline_k_new.consumer_release(smem_pipe_read);
            // if (thread_idx == 0) { print_tensor(tKpK); printf("\n"); printf("seqlen_limit = %d\n", seqlen_k_new - n_block * kBlockN);}
        };

        auto store_V = [&] (int const n_block, auto const& smem_pipe_read) {
            pipeline_v_new.consumer_wait(smem_pipe_read);
            int const n_limit = std::min(seqlen_k_new - n_block * kBlockN, kBlockN);
            Tensor tVsV_cur = tVsV(_, _, _, smem_pipe_read.index());
            if constexpr (!PagedKVNonTMA) {
                Tensor tVgV_cur = tVgV(_, _, _, n_block);
                // Clear_OOB_K must be false since we don't want to write zeros to gmem
                flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                    gmem_tiled_copy_kv, tVsV_cur, tVgV_cur, tVcV, tVpV, n_limit);
            } else {
                paged_kv_manager.store_V(n_block, tVsV_cur);
            }
            cutlass::arch::fence_view_async_shared();
            cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + flash::canonical_warp_group_idx_nosync() /*id*/);
            pipeline_v_new.consumer_release(smem_pipe_read);
        };

        #pragma unroll 1
        for (int n_block = n_block_new_max - 1; n_block >= n_block_new_min; --n_block) {
            if constexpr (PagedKVNonTMA) { paged_kv_manager.template load_page_table<true /*Seqlenk_mask*/>(n_block); }
            store_K(n_block, smem_pipe_read);
            // if (thread_idx == 0) { printf("Done storing K, n_block = %d, n_block_new_min = %d\n", n_block, n_block_new_min); }
            store_V(n_block, smem_pipe_read);
            // if (thread_idx == 0) { printf("Done storing V, n_block = %d, n_block_new_min = %d\n", n_block, n_block_new_min); }
            ++smem_pipe_read;
        }
        // if (thread_idx == 0) { printf("After for loop\n"); }

        // Re-signaling the NamedBarrier that we "canceled out"
        if constexpr (UseSchedulerBarrier) {
            if (flash::canonical_warp_group_idx_nosync() == 1) {
                cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) /*id*/);
            }
        }

        return true;

    }

};

} // namespace flash

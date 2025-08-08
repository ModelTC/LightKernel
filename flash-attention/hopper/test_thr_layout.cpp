#include <cute/config.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cute/atom/mma_atom.hpp>
#include <iostream>

using namespace cute;

int main() {
    // 创建一个具体的MMA Atom示例
    using AtomLayoutPV = Layout<Shape<_1, Int<2>, _1>>
    using TiledMmaPV = decltype(cute::make_tiled_mma(
        std::conditional_t<
            !MmaPV_is_RS,
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum,
                     TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>()),
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum,
                     TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>())
        >{},
        AtomLayoutPV{}));
    // 使用SM80_16x8x8_F16F16F16F16_TN作为例子
    using MMA_Op = SM80_16x8x8_F16F16F16F16_TN;
    auto mma_atom = MMA_Atom<MMA_Op>{};
    
    std::cout << "=== MMA Atom 信息 ===" << std::endl;
    std::cout << "Shape_MNK: ";
    print(mma_atom.Shape_MNK{});
    std::cout << std::endl;
    
    std::cout << "ThrID: ";
    print(mma_atom.ThrID{});
    std::cout << std::endl;
    
    // 创建一个简单的 TiledMMA
    auto tiled_mma = make_tiled_mma(mma_atom);
    
    std::cout << "\n=== TiledMMA 信息 ===" << std::endl;
    std::cout << "ThrLayoutVMNK: ";
    print(tiled_mma.get_thr_layout_vmnk());
    std::cout << std::endl;
    
    // 尝试不同的 tiling 配置
    auto tiled_1x2x1 = make_tiled_mma(mma_atom, Layout<Shape<_1,_2,_1>>{});
    std::cout << "\nTiling (1,2,1) - ThrLayoutVMNK: ";
    print(tiled_1x2x1.get_thr_layout_vmnk());
    std::cout << std::endl;
    
    auto tiled_2x1x1 = make_tiled_mma(mma_atom, Layout<Shape<_2,_1,_1>>{});
    std::cout << "\nTiling (2,1,1) - ThrLayoutVMNK: ";
    print(tiled_2x1x1.get_thr_layout_vmnk());
    std::cout << std::endl;
    
    return 0;
}
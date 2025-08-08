// 深入解析 get_slice(64) 的完整执行过程
// 基于我们之前分析的 ThrLayoutVMNK = Layout<(128,1,1,1), (1,0,0,0)>

#include <iostream>
#include <tuple>

// 模拟 CuTe 的基本概念
template<int... Values>
struct IntTuple {
    static constexpr int size = sizeof...(Values);
    template<int Index>
    static constexpr int get() {
        constexpr int values[] = {Values...};
        return values[Index];
    }
    
    void print() const {
        constexpr int values[] = {Values...};
        std::cout << "(";
        for (int i = 0; i < size; ++i) {
            std::cout << values[i];
            if (i < size - 1) std::cout << ",";
        }
        std::cout << ")";
    }
};

// 模拟坐标转换函数
template<int idx, int shape, int stride>
constexpr int idx2crd_single() {
    if constexpr (shape == 1) {
        // 对于 shape=1 的情况，跳过可能的 stride-0 除法
        return 0;
    } else if constexpr (stride == 0) {
        // stride=0 的特殊情况，直接返回0
        return 0;
    } else {
        return (idx / stride) % shape;
    }
}

void demonstrate_get_slice_64() {
    std::cout << "=== get_slice(64) 深入分析 ===\n\n";
    
    // 回顾我们的布局
    std::cout << "回顾 ThrLayoutVMNK 布局:\n";
    std::cout << "从 tiled_product(Layout<(128), (1)>, Layout<(1,1,1), (0,0,0)>) 得到:\n";
    std::cout << "ThrLayoutVMNK = Layout<(128,1,1,1), (1,0,0,0)>\n\n";
    
    std::cout << "布局解释:\n";
    std::cout << "  shape =  (128, 1, 1, 1)  // V维128个，M/N/K维各1个\n";
    std::cout << "  stride = (1,   0, 0, 0)  // V维步长1，M/N/K维步长0\n\n";
    
    // 步骤1: get_slice 函数调用
    std::cout << "=== 步骤1: get_slice(64) 函数调用 ===\n";
    std::cout << "auto get_slice(ThrIdx const& thr_idx) const {\n";
    std::cout << "  // 将平坦的线程索引转换为(V,M,N,K)坐标\n";
    std::cout << "  auto thr_vmnk = thr_layout_vmnk_.get_flat_coord(thr_idx);\n";
    std::cout << "  return ThrMMA<TiledMMA, decltype(thr_vmnk)>{*this, thr_vmnk};\n";
    std::cout << "}\n\n";
    
    // 步骤2: get_flat_coord 调用链
    std::cout << "=== 步骤2: get_flat_coord 调用链 ===\n";
    std::cout << "thr_layout_vmnk_.get_flat_coord(64) 调用:\n";
    std::cout << "1. get_flat_coord(64) -> crd2crd(get_hier_coord(64), shape(), repeat<rank>(Int<1>{}));\n";
    std::cout << "2. get_hier_coord(64) -> idx2crd(64, shape(), stride());\n";
    std::cout << "3. idx2crd(64, (128,1,1,1), (1,0,0,0));\n\n";
    
    // 步骤3: idx2crd 的详细计算
    std::cout << "=== 步骤3: idx2crd 的详细计算 ===\n";
    std::cout << "idx2crd 对每个维度独立计算坐标，公式: (idx / stride) % shape\n\n";
    
    const int thread_id = 64;
    
    // V 维度计算
    std::cout << "V维度计算:\n";
    std::cout << "  shape[0] = 128, stride[0] = 1\n";
    std::cout << "  coord[0] = (64 / 1) % 128 = 64 % 128 = 64\n\n";
    
    // M 维度计算  
    std::cout << "M维度计算:\n";
    std::cout << "  shape[1] = 1, stride[1] = 0\n";
    std::cout << "  由于 shape=1，直接返回 0 (跳过 stride-0 除法)\n";
    std::cout << "  coord[1] = 0\n\n";
    
    // N 维度计算
    std::cout << "N维度计算:\n";
    std::cout << "  shape[2] = 1, stride[2] = 0\n"; 
    std::cout << "  由于 shape=1，直接返回 0\n";
    std::cout << "  coord[2] = 0\n\n";
    
    // K 维度计算
    std::cout << "K维度计算:\n";
    std::cout << "  shape[3] = 1, stride[3] = 0\n";
    std::cout << "  由于 shape=1，直接返回 0\n"; 
    std::cout << "  coord[3] = 0\n\n";
    
    // 计算结果
    constexpr int v_coord = idx2crd_single<64, 128, 1>();
    constexpr int m_coord = idx2crd_single<64, 1, 0>();
    constexpr int n_coord = idx2crd_single<64, 1, 0>();
    constexpr int k_coord = idx2crd_single<64, 1, 0>();
    
    std::cout << "=== 步骤4: 坐标转换结果 ===\n";
    std::cout << "get_hier_coord(64) 返回: (" << v_coord << "," << m_coord << "," << n_coord << "," << k_coord << ")\n";
    std::cout << "get_flat_coord(64) 进一步处理但结果相同: (" << v_coord << "," << m_coord << "," << n_coord << "," << k_coord << ")\n\n";
    
    // 步骤5: ThrMMA 对象创建
    std::cout << "=== 步骤5: ThrMMA 对象创建 ===\n";
    std::cout << "return ThrMMA<TiledMMA, decltype(thr_vmnk)>{*this, thr_vmnk};\n\n";
    std::cout << "创建的 ThrMMA 对象包含:\n";
    std::cout << "- 父 TiledMMA 的所有信息和方法\n";
    std::cout << "- thr_vmnk_ = (" << v_coord << "," << m_coord << "," << n_coord << "," << k_coord << ") // 该线程的坐标\n\n";
    
    // 步骤6: ThrMMA 的实际用途
    std::cout << "=== 步骤6: ThrMMA 的实际用途 ===\n";
    std::cout << "线程64的ThrMMA对象可以用于:\n\n";
    
    std::cout << "1. partition_C(ctensor):\n";
    std::cout << "   auto thr_vmn = make_coord(get<0>(thr_vmnk_), make_coord(get<1>(thr_vmnk_), get<2>(thr_vmnk_)));\n";
    std::cout << "   -> thr_vmn = make_coord(64, make_coord(0, 0))\n";
    std::cout << "   -> thr_vmn = (64, (0, 0))\n";
    std::cout << "   选择该线程负责的C矩阵数据分片\n\n";
    
    std::cout << "2. partition_A(atensor):\n";
    std::cout << "   auto thr_vmk = make_coord(get<0>(thr_vmnk_), make_coord(get<1>(thr_vmnk_), get<3>(thr_vmnk_)));\n";
    std::cout << "   -> thr_vmk = make_coord(64, make_coord(0, 0))\n";
    std::cout << "   -> thr_vmk = (64, (0, 0))\n";
    std::cout << "   选择该线程负责的A矩阵数据分片\n\n";
    
    std::cout << "3. partition_B(btensor):\n";
    std::cout << "   auto thr_vnk = make_coord(get<0>(thr_vmnk_), make_coord(get<2>(thr_vmnk_), get<3>(thr_vmnk_)));\n";
    std::cout << "   -> thr_vnk = make_coord(64, make_coord(0, 0))\n";  
    std::cout << "   -> thr_vnk = (64, (0, 0))\n";
    std::cout << "   选择该线程负责的B矩阵数据分片\n\n";
    
    // 总结
    std::cout << "=== 总结 ===\n";
    std::cout << "对于线程ID=64:\n";
    std::cout << "1. 在VMNK空间中的坐标: (64, 0, 0, 0)\n";
    std::cout << "2. V=64 表示这是第64个参与MMA的线程\n";
    std::cout << "3. M=N=K=0 表示在MNK维度上都是第0个分块\n";
    std::cout << "4. 由于AtomLayoutMNK=(1,1,1)，只有一个分块，所以所有线程在MNK维度上坐标都是0\n";
    std::cout << "5. 不同线程只在V维度上不同，实现了线程在原子MMA内的分工\n";
}

int main() {
    demonstrate_get_slice_64();
    return 0;
} 
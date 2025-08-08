// 深入解析 tiled_product(Layout<_128>, Layout<1,1,1>) 的工作原理
// 这是一个概念性演示，展示了每一步的变换过程

#include <iostream>
#include <string>

// 模拟 CuTe 的基本概念
struct MockLayout {
    std::string shape;
    std::string stride;
    int rank;
    
    MockLayout(std::string s, std::string st, int r) : shape(s), stride(st), rank(r) {}
    
    void print() const {
        std::cout << "Layout<" << shape << ", " << stride << "> (rank=" << rank << ")\n";
    }
};

void demonstrate_tiled_product() {
    std::cout << "=== tiled_product 深入分析 ===\n\n";
    
    // 输入参数
    std::cout << "输入参数:\n";
    std::cout << "AtomThrID    = Layout<(128), (1)>     // 1维, 128个线程\n";
    std::cout << "AtomLayoutMNK = Layout<(1,1,1), (0,0,0)> // 3维, 每维大小1\n\n";
    
    // 步骤1: tiled_product 调用 zipped_product
    std::cout << "=== 步骤1: tiled_product 调用 zipped_product ===\n";
    std::cout << "tiled_product(block, tiler) {\n";
    std::cout << "  auto result = zipped_product(block, tiler);\n";
    std::cout << "  auto R1 = rank<1>(result);  // 获取第二个模式的rank\n";
    std::cout << "  return result(_, repeat<R1>(_));  // 解包第二个模式\n";
    std::cout << "}\n\n";
    
    // 步骤2: zipped_product 调用 logical_product 和 tile_unzip
    std::cout << "=== 步骤2: zipped_product 的工作 ===\n";
    std::cout << "zipped_product(block, tiler) {\n";
    std::cout << "  return tile_unzip(logical_product(block, tiler), tiler);\n";
    std::cout << "}\n\n";
    
    // 步骤3: logical_product 创建复合布局
    std::cout << "=== 步骤3: logical_product 创建复合布局 ===\n";
    std::cout << "logical_product(Layout<(128), (1)>, Layout<(1,1,1), (0,0,0)>) {\n";
    std::cout << "  // 创建一个复合布局，将 block 作为第一部分\n";
    std::cout << "  // 将 tiler 通过 complement 变换作为第二部分\n";
    std::cout << "  return make_layout(\n";
    std::cout << "    Layout<(128), (1)>,                    // 第一部分: 原始block\n";
    std::cout << "    composition(complement(...), tiler)    // 第二部分: 变换后的tiler\n";
    std::cout << "  );\n";
    std::cout << "}\n";
    std::cout << "结果: Layout<((128), (1,1,1)), ((1), (0,0,0))>  // rank=2\n\n";
    
    // 步骤4: tile_unzip 重新组织结构
    std::cout << "=== 步骤4: tile_unzip 重新组织结构 ===\n";
    std::cout << "tile_unzip(..., Layout<(1,1,1), (0,0,0)>) {\n";
    std::cout << "  // 使用 zip2_by 按照 tiler 的结构重新组织\n";
    std::cout << "  return make_layout(\n";
    std::cout << "    zip2_by(shape,  tiler),   // 重组shape\n";
    std::cout << "    zip2_by(stride, tiler)    // 重组stride\n";
    std::cout << "  );\n";
    std::cout << "}\n\n";
    
    // 步骤5: zip2_by 的具体工作
    std::cout << "=== 步骤5: zip2_by 的具体工作 ===\n";
    std::cout << "zip2_by 会根据 tiler 的结构 (1,1,1) 来组织数据:\n";
    std::cout << "原始shape:  ((128), (1,1,1))\n";
    std::cout << "tiler结构:  (1,1,1)          // 3个维度\n";
    std::cout << "zip2_by结果: ((128,1), (128,1), (128,1))\n";
    std::cout << "           // 每个tiler维度都与block的128配对\n\n";
    
    // 步骤6: 最终的解包操作
    std::cout << "=== 步骤6: 最终解包操作 ===\n";
    std::cout << "zipped_product 产生了:\n";
    std::cout << "Layout<((128,1), (128,1), (128,1)), ((1,0), (1,0), (1,0))>\n\n";
    
    std::cout << "tiled_product 进行解包:\n";
    std::cout << "auto R1 = rank<1>(result) = 3;  // 第二模式有3个子模式\n";
    std::cout << "result(_, repeat<3>(_))\n";
    std::cout << "= result(_, (_, _, _))  // 展开第二个模式的所有子模式\n\n";
    
    // 最终结果
    std::cout << "=== 最终结果分析 ===\n";
    std::cout << "ThrLayoutVMNK = Layout<(128,1,1,1), (1,0,0,0)>\n\n";
    std::cout << "这个布局的含义:\n";
    std::cout << "- 总共4个维度: (V, M, N, K)\n";
    std::cout << "- V维度: 128个线程 (来自AtomThrID)\n";
    std::cout << "- M维度: 1个分块 (来自AtomLayoutMNK的第1维)\n";
    std::cout << "- N维度: 1个分块 (来自AtomLayoutMNK的第2维)\n";
    std::cout << "- K维度: 1个分块 (来自AtomLayoutMNK的第3维)\n\n";
    
    std::cout << "步长 (1,0,0,0) 的含义:\n";
    std::cout << "- V维度步长=1: 线程索引连续递增\n";
    std::cout << "- M,N,K维度步长=0: 所有线程在这些维度上索引相同\n\n";
    
    std::cout << "=== 映射关系 ===\n";
    std::cout << "线程索引映射: (v,m,n,k) -> thread_idx\n";
    std::cout << "例如:\n";
    std::cout << "  线程0: (0,0,0,0) -> 0\n";
    std::cout << "  线程1: (1,0,0,0) -> 1\n";
    std::cout << "  线程127: (127,0,0,0) -> 127\n";
    std::cout << "所有线程在M,N,K维度上的坐标都是(0,0,0)\n";
}

int main() {
    demonstrate_tiled_product();
    return 0;
} 
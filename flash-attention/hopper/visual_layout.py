#!/usr/bin/env python3

print("=== Layout Transformation Visualization ===\n")

print("Original Layout: ((_2,_2,_8),_1,_1):((_1,_2,_4),_0,_0)")
print("- Shape: ((2,2,8), 1, 1) = 2×2×8×1×1 = 32 elements")
print("- Stride: ((1,2,4), 0, 0)")
print("- Interpretation: 3D tensor with MMA-style layout\n")

print("Converted Layout: ((_2,_1),(_2,_8,_1)):((_2,_0),(_1,_4,_0))")
print("- Shape: ((2,1), (2,8,1)) = 2×1 rows × 2×8×1 cols = 2×16 = 32 elements") 
print("- Stride: ((2,0), (1,4,0))")
print("- Interpretation: Row-Column format for easier masking\n")

# 创建32个元素的映射表
elements = {}
for addr in range(32):
    elements[addr] = {'orig': None, 'conv': None, 'row': None, 'col': None}

# 计算原始layout的索引
print("=== Original Layout Element Mapping ===")
for i0 in range(2):
    for i1 in range(2):
        for i2 in range(8):
            for i3 in range(1):
                for i4 in range(1):
                    addr = i0*1 + i1*2 + i2*4 + i3*0 + i4*0
                    elements[addr]['orig'] = (i0,i1,i2,i3,i4)

# 计算转换后layout的索引和行列位置
print("=== Converted Layout Element Mapping ===")
for r0 in range(2):      # Row dimension part 1
    for r1 in range(1):  # Row dimension part 2
        for c0 in range(2):      # Col dimension part 1
            for c1 in range(8):  # Col dimension part 2
                for c2 in range(1):  # Col dimension part 3
                    addr = r0*2 + r1*0 + c0*1 + c1*4 + c2*0
                    elements[addr]['conv'] = (r0,r1,c0,c1,c2)
                    # 计算实际的行列位置
                    row = r0 * 1 + r1  # 2*r0 + r1，但r1总是0
                    col = c0 * 8 * 1 + c1 * 1 + c2  # 8*c0 + c1，因为c2总是0
                    elements[addr]['row'] = row
                    elements[addr]['col'] = col

# 打印完整的映射关系
print("\n=== Complete Element Mapping ===")
print("Addr | Original Index     | Converted Index    | Row | Col | Pattern")
print("-----|--------------------|--------------------|-----|-----|--------")
for addr in sorted(elements.keys()):
    orig = elements[addr]['orig']
    conv = elements[addr]['conv'] 
    row = elements[addr]['row']
    col = elements[addr]['col']
    print(f"{addr:4d} | {str(orig):18s} | {str(conv):18s} | {row:3d} | {col:3d} | {addr%4}")

print("\n=== Row-Column Matrix Visualization ===")
print("Showing how elements are arranged in the converted 2×16 row-col format:")
print("(Numbers show the linear memory addresses)\n")

# 创建2x16的矩阵来显示地址分布
matrix = [[-1 for _ in range(16)] for _ in range(2)]
for addr in range(32):
    row = elements[addr]['row']
    col = elements[addr]['col']
    matrix[row][col] = addr

print("    Col:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15")
print("         +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+")
for row in range(2):
    print(f"Row {row}: |", end="")
    for col in range(16):
        print(f"{matrix[row][col]:2d}|", end="")
    print()

print("\n=== Access Pattern Analysis ===")
print("Notice the interleaved pattern:")
print("- Row 0: 0,4,8,12,16,20,24,28, 1,5,9,13,17,21,25,29")
print("- Row 1: 2,6,10,14,18,22,26,30, 3,7,11,15,19,23,27,31")
print("\nThis shows how the MMA layout is transformed into row-column format")
print("while preserving the memory access pattern for efficient computation.")

print("\n=== Key Insights ===")
print("1. The transformation reorganizes the same 32 elements")
print("2. Memory addresses remain the same, but indexing changes")
print("3. Row-column format makes masking operations more intuitive")
print("4. Each 'row' in the converted format represents different thread groups")
print("5. Each 'column' represents different positions in the attention matrix") 
#!/usr/bin/env python3

print("=== Causal Mask 可视化示例 ===\n")

# 参数设置
seqlen = 10  # 简化的序列长度
block_size = 4  # 简化的块大小

print(f"序列长度: {seqlen}")
print(f"块大小: {block_size}")
print(f"总共需要 {(seqlen + block_size - 1) // block_size} 个Key块")

print(f"\n=== 完整的Causal Mask矩阵 ===")
print("(1表示可见，0表示被mask)")
print("    Key位置:", end="")
for k in range(seqlen):
    print(f"{k:2d}", end="")
print()

for q in range(seqlen):
    print(f"Query{q:2d}:   ", end="")
    for k in range(seqlen):
        if k <= q:  # 因果mask：只能看到之前的位置
            print(" 1", end="")
        else:
            print(" 0", end="")
    print()

print(f"\n=== 分块处理过程 ===")

num_key_blocks = (seqlen + block_size - 1) // block_size

for n_block in range(num_key_blocks):
    key_start = n_block * block_size
    key_end = min(key_start + block_size - 1, seqlen - 1)
    
    print(f"\n--- 处理Key块 {n_block} [位置 {key_start}-{key_end}] ---")
    
    # 计算causal_row_offset
    causal_row_offset = 1 + seqlen - n_block * block_size - seqlen - 0
    print(f"causal_row_offset = 1 + {seqlen} - {n_block} * {block_size} - {seqlen} - 0 = {causal_row_offset}")
    
    print(f"\n当前Key块内的mask模式:")
    print("      Key块内位置:", end="")
    for k_rel in range(min(block_size, seqlen - key_start)):
        print(f"{k_rel:2d}", end="")
    print()
    
    for q in range(seqlen):
        # 计算当前Query能看到的右边界
        col_limit_right = q + causal_row_offset
        
        print(f"Query{q:2d} (边界{col_limit_right:2d}): ", end="")
        
        for k_rel in range(min(block_size, seqlen - key_start)):
            k_global = key_start + k_rel
            
            # 判断是否可见
            if k_global <= q and k_rel < col_limit_right:
                print(" 1", end="")
            else:
                print(" 0", end="")
        print(f"  # Query{q}全局可见到{q}, 块内限制{col_limit_right}")

print(f"\n=== 关键理解 ===")
print("1. causal_row_offset 是一个'坐标转换偏移'")
print("2. 它将全局的因果关系转换为当前Key块内的相对位置")
print("3. row_idx + causal_row_offset 给出了当前Query在当前Key块中的可见右边界")
print("4. 负的causal_row_offset表示大部分Query看不到当前Key块（Key块在Query之后）")
print("5. 正的causal_row_offset表示大部分Query可以看到当前Key块的一部分")

print(f"\n=== 实际代码中的应用 ===")
print("在mask.h中：")
print("```cpp")
print("int const col_limit_right = row_idx + causal_row_offset;")
print("for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {")
print("    if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= col_limit_right) {")
print("        tSrS_rowcol(m, n) = -INFINITY;  // mask掉")
print("    }")
print("}")
print("```")
print("这里 get<Col>(t0ScS_rowcol(_0{}, n)) 获取的是Key块内的相对列位置，")
print("与 col_limit_right 比较来决定是否mask。") 
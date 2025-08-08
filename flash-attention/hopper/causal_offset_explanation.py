#!/usr/bin/env python3

print("=== Causal Row Offset 详细解释 ===\n")

print("公式：causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q - thread_col_offset\n")

# 示例场景
seqlen_k = 100      # Key序列长度
seqlen_q = 100      # Query序列长度（通常等于seqlen_k）
n_block = 1         # 当前处理的Key块索引
kBlockN = 64        # 每个块的大小
thread_col_offset = 0  # 当前线程的列偏移

causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q - thread_col_offset

print(f"示例参数：")
print(f"  seqlen_k = {seqlen_k} (Key序列长度)")
print(f"  seqlen_q = {seqlen_q} (Query序列长度)")
print(f"  n_block = {n_block} (当前Key块索引)")
print(f"  kBlockN = {kBlockN} (块大小)")
print(f"  thread_col_offset = {thread_col_offset} (线程列偏移)")
print(f"\ncausal_row_offset = 1 + {seqlen_k} - {n_block} * {kBlockN} - {seqlen_q} - {thread_col_offset}")
print(f"                  = 1 + {seqlen_k} - {n_block * kBlockN} - {seqlen_q} - {thread_col_offset}")
print(f"                  = {causal_row_offset}")

print(f"\n=== Causal Mask 的基本原理 ===")
print("在因果注意力中，位置i的Query只能看到位置0到i的Key：")
print("  Query[0] 可以看到 Key[0]")
print("  Query[1] 可以看到 Key[0, 1]")
print("  Query[2] 可以看到 Key[0, 1, 2]")
print("  ...")
print("  Query[i] 可以看到 Key[0, 1, 2, ..., i]")

print(f"\n=== 为什么需要 causal_row_offset ===")
print("在分块计算中，我们需要确定当前Key块中哪些位置对哪些Query位置是可见的。")
print("causal_row_offset 定义了一个'基准偏移'，用于计算每个Query位置的可见边界。")

print(f"\n=== 具体计算过程 ===")
print("对于Query位置 row_idx，它能看到的Key位置的右边界是：")
print("  col_limit_right = row_idx + causal_row_offset")
print(f"  col_limit_right = row_idx + {causal_row_offset}")

print(f"\n让我们看几个具体的Query位置：")

# 模拟几个Query位置
current_key_block_start = n_block * kBlockN  # 当前Key块的起始位置
print(f"\n当前Key块范围: [{current_key_block_start}, {current_key_block_start + kBlockN - 1}]")

for query_pos in [35, 36, 37, 63, 64, 65, 100]:
    if query_pos >= seqlen_q:
        continue
    
    # 计算这个Query位置能看到的Key位置右边界（全局坐标）
    global_col_limit = query_pos + 1  # 因果mask：位置i能看到0到i
    
    # 转换为当前Key块内的相对位置
    relative_col_limit = global_col_limit - current_key_block_start
    
    # 使用公式计算的结果
    formula_result = query_pos + causal_row_offset
    
    print(f"Query位置 {query_pos:3d}:")
    print(f"  - 全局可见范围: [0, {query_pos}]")
    print(f"  - 当前Key块内的相对限制: {relative_col_limit}")
    print(f"  - 公式计算结果: {query_pos} + {causal_row_offset} = {formula_result}")
    print(f"  - 匹配: {'✓' if relative_col_limit == formula_result else '✗'}")

print(f"\n=== 公式各项的含义 ===")
print(f"causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q - thread_col_offset")
print(f"                  = {1} + {seqlen_k} - {n_block * kBlockN} - {seqlen_q} - {thread_col_offset}")

print(f"\n各项解释：")
print(f"• 1: 因为位置i能看到位置0到i（包含i），所以是i+1个位置")
print(f"• seqlen_k ({seqlen_k}): Key序列的总长度")
print(f"• n_block * kBlockN ({n_block * kBlockN}): 当前Key块的起始位置")
print(f"• seqlen_q ({seqlen_q}): Query序列长度（通常等于seqlen_k）")
print(f"• thread_col_offset ({thread_col_offset}): 线程内的列偏移")

print(f"\n=== 为什么要加上 row_idx ===")
print("最终的列限制计算：col_limit_right = row_idx + causal_row_offset")
print("这是因为：")
print("1. row_idx 是当前Query的位置")
print("2. causal_row_offset 是一个'基准偏移'，将全局位置转换为当前Key块的相对位置")
print("3. 两者相加得到当前Query在当前Key块中能看到的最右边界")

print(f"\n=== 边界情况分析 ===")
# 分析不同n_block的情况
print("不同Key块的causal_row_offset：")
for block_idx in range(3):
    offset = 1 + seqlen_k - block_idx * kBlockN - seqlen_q - thread_col_offset
    block_start = block_idx * kBlockN
    block_end = min(block_start + kBlockN - 1, seqlen_k - 1)
    print(f"  Key块 {block_idx} [位置 {block_start:2d}-{block_end:2d}]: causal_row_offset = {offset}") 
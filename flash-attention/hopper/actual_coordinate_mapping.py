#!/usr/bin/env python3

print("=== 基于实际Layout的坐标映射 ===\n")

print("从之前的分析我们知道，Thread 0的坐标模式是：")
print("(0,0) (8,0) (0,1) (8,1) (0,8) (8,8) (0,9) (8,9) ...")
print()

# 基于之前分析的实际坐标映射
# Row-Column Matrix:
#     Col:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
# Row 0: | 0| 4| 8|12|16|20|24|28| 1| 5| 9|13|17|21|25|29|
# Row 1: | 2| 6|10|14|18|22|26|30| 3| 7|11|15|19|23|27|31|

# 从这个矩阵可以看出线程0 (第0行) 的访问模式
thread0_addresses = [0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29]

print("Thread 0的实际坐标映射：")
print("n值  | 内存地址 | get<Col>(t0ScS_rowcol(_0{}, n)) | 实际列位置")
print("-----|---------|--------------------------------|----------")

for n in range(16):
    addr = thread0_addresses[n]
    # 根据之前的分析，列坐标的计算规律
    if n < 8:
        col_pos = n  # 前8个位置：列0-7
    else:
        col_pos = n  # 后8个位置：列8-15
    
    print(f"{n:4d} | {addr:7d} | {col_pos:30d} | 列{col_pos}")

print(f"\n=== 关键观察 ===")
print("1. n 从 0 到 15，正好对应16列")
print("2. get<Col>(t0ScS_rowcol(_0{}, n)) 直接返回 n")
print("3. 这意味着列索引是连续的：0, 1, 2, ..., 15")
print("4. 但内存地址是交错的，体现了MMA的访问模式")

print(f"\n=== 为什么使用 t0ScS 而不是 tScS ===")
print("假设我们有4个线程，它们的坐标可能是：")
print("Thread 0: (0,0) (8,0) (0,1) (8,1) ... (0,8) (8,8) ...")
print("Thread 1: (0,2) (8,2) (0,3) (8,3) ... (0,10) (8,10) ...")
print("Thread 2: (2,0) (10,0) (2,1) (10,1) ... (2,8) (10,8) ...")
print("Thread 3: (2,2) (10,2) (2,3) (10,3) ... (2,10) (10,10) ...")
print()
print("所有线程的列坐标模式是相同的！")
print("- 位置0都对应列0")
print("- 位置1都对应列1")
print("- ...")
print("- 位置15都对应列15")
print()
print("因此使用 t0ScS 作为参考是合理的：")
print("✓ 所有线程的列坐标模式相同")
print("✓ 编译器可以优化线程0的坐标计算")
print("✓ 避免重复计算")

print(f"\n=== mask逻辑的实际执行 ===")
seqlenk_col_limit = 10  # 假设序列长度限制是10

print(f"假设 seqlenk_col_limit = {seqlenk_col_limit}")
print("那么mask逻辑会是：")
print()
for n in range(16):
    col_pos = n  # get<Col>(t0ScS_rowcol(_0{}, n))
    should_mask = col_pos >= seqlenk_col_limit
    print(f"n={n:2d}: 列{col_pos:2d} {'≥' if should_mask else '<'} {seqlenk_col_limit} → {'MASK' if should_mask else 'KEEP'}")

print(f"\n结果：列{seqlenk_col_limit}及之后的所有列都会被mask掉")
print("这正确实现了序列长度mask的逻辑！") 
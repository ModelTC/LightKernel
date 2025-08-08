#!/usr/bin/env python3

def compute_layout_coordinates(shape, stride):
    """计算给定shape和stride的所有坐标"""
    def flatten_shape(s):
        if isinstance(s, tuple):
            result = []
            for item in s:
                if isinstance(item, tuple):
                    result.extend(flatten_shape(item))
                else:
                    result.append(item)
            return result
        return [s]
    
    def flatten_stride(s):
        if isinstance(s, tuple):
            result = []
            for item in s:
                if isinstance(item, tuple):
                    result.extend(flatten_stride(item))
                else:
                    result.append(item)
            return result
        return [s]
    
    flat_shape = flatten_shape(shape)
    flat_stride = flatten_stride(stride)
    
    print(f"Flattened shape: {flat_shape}")
    print(f"Flattened stride: {flat_stride}")
    
    # 生成所有可能的索引组合
    coordinates = []
    indices = [0] * len(flat_shape)
    
    def generate_all_indices(pos):
        if pos == len(flat_shape):
            # 计算线性索引
            linear_idx = sum(indices[i] * flat_stride[i] for i in range(len(indices)))
            coordinates.append((tuple(indices.copy()), linear_idx))
            return
        
        for i in range(flat_shape[pos]):
            indices[pos] = i
            generate_all_indices(pos + 1)
    
    generate_all_indices(0)
    return coordinates

print("=== Original tSrS Layout Analysis ===")
print("Layout: ((_2,_2,_8),_1,_1):((_1,_2,_4),_0,_0)")
print("Shape: ((2,2,8), 1, 1)")
print("Stride: ((1,2,4), 0, 0)")

original_shape = ((2,2,8), 1, 1)
original_stride = ((1,2,4), 0, 0)

original_coords = compute_layout_coordinates(original_shape, original_stride)
print(f"\nTotal elements: {len(original_coords)}")
print("First 16 elements (index -> linear_address):")
for i, (idx, addr) in enumerate(original_coords[:16]):
    print(f"  {idx} -> {addr}")

print("\n=== Converted tSrS_rowcol Layout Analysis ===")
print("Layout: ((_2,_1),(_2,_8,_1)):((_2,_0),(_1,_4,_0))")
print("Shape: ((2,1), (2,8,1))")
print("Stride: ((2,0), (1,4,0))")

converted_shape = ((2,1), (2,8,1))
converted_stride = ((2,0), (1,4,0))

converted_coords = compute_layout_coordinates(converted_shape, converted_stride)
print(f"\nTotal elements: {len(converted_coords)}")
print("First 16 elements (index -> linear_address):")
for i, (idx, addr) in enumerate(converted_coords[:16]):
    print(f"  {idx} -> {addr}")

print("\n=== Mapping Analysis ===")
print("Comparing how the same linear addresses map to different indices:")

# 创建地址到索引的映射
orig_addr_to_idx = {addr: idx for idx, addr in original_coords}
conv_addr_to_idx = {addr: idx for idx, addr in converted_coords}

# 找到共同的地址
common_addrs = sorted(set(orig_addr_to_idx.keys()) & set(conv_addr_to_idx.keys()))

print(f"\nCommon addresses: {len(common_addrs)}")
print("Address -> Original_Index -> Converted_Index")
for addr in common_addrs[:16]:
    orig_idx = orig_addr_to_idx[addr]
    conv_idx = conv_addr_to_idx[addr]
    print(f"  {addr:2d} -> {orig_idx} -> {conv_idx}")

print("\n=== Row-Column Interpretation ===")
print("In the converted layout ((_2,_1),(_2,_8,_1)):")
print("- First part (_2,_1): Row dimension (2 rows, 1 sub-row)")
print("- Second part (_2,_8,_1): Column dimension (2x8x1 = 16 columns)")
print("\nElement access pattern in row-col format:")
for addr in common_addrs[:16]:
    conv_idx = conv_addr_to_idx[addr]
    # 解析为 (row_indices, col_indices)
    row_part = conv_idx[:2]  # (2,1)
    col_part = conv_idx[2:]  # (2,8,1)
    
    # 计算实际的行和列索引
    row_idx = row_part[0] * 1 + row_part[1]  # 2*0 + 0 = 0, etc.
    col_idx = col_part[0] * 8 * 1 + col_part[1] * 1 + col_part[2]
    
    print(f"  Address {addr:2d}: {conv_idx} -> Row {row_idx}, Col {col_idx}") 
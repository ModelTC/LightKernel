ARG CUDA_VERSION=12.6.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04
ARG PYTHON_VERSION=3.10
ARG MAMBA_VERSION=24.7.1-0
ARG TARGETPLATFORM

ENV PATH=/opt/conda/bin:$PATH \
    CONDA_PREFIX=/opt/conda

# Install system dependencies
RUN chmod 777 -R /tmp && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl-dev \
    curl \
    g++ \
    make \
    git \
    cmake \
    ninja-build \
    build-essential \
    ccache && \
    rm -rf /var/lib/apt/lists/*

# Install Mambaforge
RUN case ${TARGETPLATFORM} in \
    "linux/arm64")  MAMBA_ARCH=aarch64  ;; \
    *)              MAMBA_ARCH=x86_64   ;; \
    esac && \
    curl -fsSL -o ~/mambaforge.sh -v "https://github.com/conda-forge/miniforge/releases/download/${MAMBA_VERSION}/Mambaforge-${MAMBA_VERSION}-Linux-${MAMBA_ARCH}.sh" && \
    bash ~/mambaforge.sh -b -p /opt/conda && \
    rm ~/mambaforge.sh

# Install Python
RUN case ${TARGETPLATFORM} in \
    "linux/arm64")  exit 1 ;; \
    *)              /opt/conda/bin/conda update -y conda &&  \
    /opt/conda/bin/conda install -y "python=${PYTHON_VERSION}" ;; \
    esac && \
    /opt/conda/bin/conda clean -ya

# Set working directory
WORKDIR /workspace

# Install PyTorch with CUDA support
RUN pip install torch==2.7.1

# Install build dependencies + 构建加速工具
RUN pip install --upgrade pip setuptools wheel build scikit-build-core[pyproject] pybind11 ninja psutil

# 🚀 设置ccache编译缓存加速
ENV CCACHE_DIR=/tmp/ccache \
    CCACHE_MAXSIZE=2G \
    CCACHE_COMPRESS=true \
    CC="ccache gcc" \
    CXX="ccache g++"
RUN ccache --set-config=max_size=2G

# Copy source code to container
COPY . .

# 🔧 设置 PyTorch 路径，让 CMake 能找到 Torch 配置
# 获取 PyTorch 安装路径并设置 CMAKE_PREFIX_PATH
RUN python -c "import torch; print(f'PyTorch installed at: {torch.__path__[0]}')" && \
    TORCH_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") && \
    echo "Torch CMAKE path: $TORCH_PATH"

# Set environment variables for building
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE \
    FLASH_ATTENTION_DISABLE_BACKWARD=TRUE \
    CUDA_HOME=/usr/local/cuda \
    CUDA_ROOT=/usr/local/cuda \
    CCACHE_DISABLE=0

# 🎯 关键修复：设置 CMAKE_PREFIX_PATH 让 CMake 找到 PyTorch
RUN TORCH_CMAKE_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") && \
    echo "export CMAKE_PREFIX_PATH=$TORCH_CMAKE_PATH:\$CMAKE_PREFIX_PATH" >> ~/.bashrc && \
    echo "CMAKE_PREFIX_PATH=$TORCH_CMAKE_PATH" >> /etc/environment

# 🚀 GitHub Actions优化：智能设置并行度（针对2核7GB限制）
RUN python -c "\
import os, psutil; \
cpu_cores = min(2, os.cpu_count()); \
available_memory_gb = min(7, psutil.virtual_memory().available / (1024**3)); \
memory_jobs = max(1, int(available_memory_gb / 3)); \
optimal_jobs = min(cpu_cores, memory_jobs, 2); \
nvcc_threads = optimal_jobs; \
print(f'🎯 CI优化: MAX_JOBS={optimal_jobs}, NVCC_THREADS={nvcc_threads}'); \
print(f'💾 估算资源: {available_memory_gb:.1f}GB, {cpu_cores}核'); \
f = open('/etc/environment', 'a'); \
f.write(f'MAX_JOBS={optimal_jobs}\n'); \
f.write(f'NVCC_THREADS={nvcc_threads}\n'); \
f.close()"

# Create output directory
RUN mkdir -p /out



# Build lightllm-kernel package (main project)  
RUN echo "🔧 Building lightllm-kernel package..." && \
    echo "📋 Verifying PyTorch installation..." && \
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CMake prefix path: {torch.utils.cmake_prefix_path}')" && \
    eval $(cat /etc/environment | xargs -I {} echo export {}) && \
    TORCH_CMAKE_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") && \
    echo "🔧 Setting CMAKE_PREFIX_PATH to: $TORCH_CMAKE_PATH" && \
    echo "🚀 Using optimized settings: MAX_JOBS=$MAX_JOBS, NVCC_THREADS=$NVCC_THREADS" && \
    CMAKE_PREFIX_PATH="$TORCH_CMAKE_PATH:$CMAKE_PREFIX_PATH" python -m build --wheel --outdir /out/ && \
    echo "✅ lightllm-kernel build completed"

# Build flash_attn_3 package (hopper) - 源码优化构建
RUN echo "🔧 Building flash_attn_3 from source with optimizations..." && \
    cd flash-attention/hopper && \
    eval $(cat /etc/environment | xargs -I {} echo export {}) && \
    echo "🚀 Optimized settings: MAX_JOBS=$MAX_JOBS, NVCC_THREADS=$NVCC_THREADS" && \
    echo "⏰ GitHub Actions: Building within 6h time limit..." && \
    MAX_JOBS=$MAX_JOBS NVCC_THREADS=$NVCC_THREADS FLASH_ATTN_CUDA_ARCHS=90 python setup.py bdist_wheel && \
    cp dist/*.whl /out/ && \
    echo "✅ flash_attn_3 optimized source build completed"

# 显示编译缓存统计（如果可用）
RUN ccache --show-stats 2>/dev/null || echo "💾 ccache stats not available"

# Verify all wheels are built (源码构建验证)
RUN echo "📦 Final wheel packages:" && \
    ls -la /out/ && \
    WHEEL_COUNT=$(ls -1 /out/*.whl | wc -l) && \
    echo "🎯 Total wheels built: $WHEEL_COUNT" && \
    if [ "$WHEEL_COUNT" -ne 2 ]; then \
        echo "❌ ERROR: Expected 2 wheels (lightllm-kernel + flash_attn_3), found $WHEEL_COUNT" && \
        echo "📋 Debug info:" && ls -la /out/ && \
        exit 1; \
    else \
        echo "🎉 SUCCESS: All wheels built from optimized source compilation!"; \
    fi && \
    echo "🕒 Optimized build completed within GitHub Actions time limit!"
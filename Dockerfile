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
    build-essential && \
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

# Install build dependencies
RUN pip install --upgrade pip setuptools wheel build scikit-build-core[pyproject] pybind11 ninja

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
    CUDA_ROOT=/usr/local/cuda

# 🎯 关键修复：设置 CMAKE_PREFIX_PATH 让 CMake 找到 PyTorch
RUN TORCH_CMAKE_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") && \
    echo "export CMAKE_PREFIX_PATH=$TORCH_CMAKE_PATH:\$CMAKE_PREFIX_PATH" >> ~/.bashrc && \
    echo "CMAKE_PREFIX_PATH=$TORCH_CMAKE_PATH" >> /etc/environment

# Create output directory
RUN mkdir -p /out

# Build lightllm-kernel package (main project)  
# 🎯 关键：在构建时设置 CMAKE_PREFIX_PATH，让 CMake 找到 PyTorch
RUN echo "🔧 Building lightllm-kernel package..." && \
    echo "📋 Verifying PyTorch installation..." && \
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CMake prefix path: {torch.utils.cmake_prefix_path}')" && \
    TORCH_CMAKE_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") && \
    echo "🔧 Setting CMAKE_PREFIX_PATH to: $TORCH_CMAKE_PATH" && \
    CMAKE_PREFIX_PATH="$TORCH_CMAKE_PATH:$CMAKE_PREFIX_PATH" python -m build --wheel --outdir /out/ && \
    echo "✅ lightllm-kernel build completed"

# Use prebuilt flash_attn_3 wheel (saves ~50+ minutes!)
RUN echo "📦 Using prebuilt flash_attn_3 wheel..." && \
    cp flash-attention/hopper/dist/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl /out/ && \
    echo "✅ flash_attn_3 wheel copied successfully"

# Fallback: Build from source if needed (uncomment if you need to rebuild)
# RUN echo "🔧 Building flash_attn_3 package..." && \
#     cd flash-attention/hopper && \
#     MAX_JOBS=2 NVCC_THREADS=2 FLASH_ATTN_CUDA_ARCHS=90 FLASH_ATTENTION_DISABLE_SM80=TRUE python setup.py bdist_wheel && \
#     cp dist/*.whl /out/ && \
#     echo "✅ flash_attn_3 build completed"

# Verify all wheels are built
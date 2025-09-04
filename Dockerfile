ARG CUDA_VERSION=12.6.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04
ARG PYTHON_VERSION=3.10
ARG MAMBA_VERSION=24.7.1-0
ARG TARGETPLATFORM

ENV PATH=/opt/conda/bin:$PATH \
    CONDA_PREFIX=/opt/conda

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

RUN case ${TARGETPLATFORM} in \
    "linux/arm64")  MAMBA_ARCH=aarch64  ;; \
    *)              MAMBA_ARCH=x86_64   ;; \
    esac && \
    curl -fsSL -o ~/mambaforge.sh -v "https://github.com/conda-forge/miniforge/releases/download/${MAMBA_VERSION}/Mambaforge-${MAMBA_VERSION}-Linux-${MAMBA_ARCH}.sh" && \
    bash ~/mambaforge.sh -b -p /opt/conda && \
    rm ~/mambaforge.sh

RUN case ${TARGETPLATFORM} in \
    "linux/arm64")  exit 1 ;; \
    *)              /opt/conda/bin/conda update -y conda &&  \
    /opt/conda/bin/conda install -y "python=${PYTHON_VERSION}" ;; \
    esac && \
    /opt/conda/bin/conda clean -ya


WORKDIR /root

RUN pip install torch==2.7.1 numpy

COPY . .

# 🔧 设置 PyTorch 路径，让 CMake 能找到 Torch 配置
# 获取 PyTorch 安装路径并设置 CMAKE_PREFIX_PATH
RUN python -c "import torch; print(f'PyTorch installed at: {torch.__path__[0]}')" && \
    TORCH_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") && \
    echo "Torch CMAKE path: $TORCH_PATH"

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_ROOT=/usr/local/cuda

# 🎯 关键修复：设置 CMAKE_PREFIX_PATH 让 CMake 找到 PyTorch
RUN TORCH_CMAKE_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") && \
    echo "export CMAKE_PREFIX_PATH=$TORCH_CMAKE_PATH:\$CMAKE_PREFIX_PATH" >> ~/.bashrc && \
    echo "CMAKE_PREFIX_PATH=$TORCH_CMAKE_PATH" >> /etc/environment

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

# Build flash_attn_3 package (hopper)
RUN echo "🔧 Building flash_attn_3 package..." && \
    cd flash-attention/hopper && \
    MAX_JOBS=1 NVCC_THREADS=1 FLASH_ATTN_CUDA_ARCHS=90 FLASH_ATTENTION_DISABLE_SM80=TRUE python setup.py bdist_wheel && \
    cp dist/*.whl /out/ && \
    echo "✅ flash_attn_3 build completed"

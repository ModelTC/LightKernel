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

RUN git clone https://github.com/ModelTC/LightKernel.git

# Set environment variables for building
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE \
    FLASH_ATTENTION_DISABLE_BACKWARD=TRUE \
    CUDA_HOME=/usr/local/cuda \
    CUDA_ROOT=/usr/local/cuda

# Create output directory
RUN mkdir -p /out

# Build lightllm-kernel package (main project)
RUN echo "🔧 Building lightllm-kernel package..." && \
    python -m build --wheel --outdir /out/ && \
    echo "✅ lightllm-kernel build completed"

# Build flash_attn_3 package (hopper)
RUN echo "🔧 Building flash_attn_3 package..." && \
    cd flash-attention/hopper && \
    MAX_JOBS=1 NVCC_THREADS=1 FLASH_ATTN_CUDA_ARCHS="80;86;89;90" python setup.py bdist_wheel && \
    cp dist/*.whl /out/ && \
    echo "✅ flash_attn_3 build completed"

# Verify all wheels are built
RUN echo "📦 Final wheel packages:" && \
    ls -la /out/ && \
    WHEEL_COUNT=$(ls -1 /out/*.whl | wc -l) && \
    echo "Total wheels built: $WHEEL_COUNT" && \
    if [ "$WHEEL_COUNT" -ne 2 ]; then \
        echo "❌ Error: Expected 2 wheels, found $WHEEL_COUNT" && exit 1; \
    else \
        echo "✅ Successfully built all wheel packages"; \
    fi 
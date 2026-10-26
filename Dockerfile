FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]

ARG VERSION_ID="0.1.3"
ARG SCLI_URL="https://scallop-lang.github.io/artifacts/scli/x86_64-linux-unknown/v${VERSION_ID}/scli"
ARG SCLREPL_URL="https://scallop-lang.github.io/artifacts/sclrepl/x86_64-linux-unknown/v${VERSION_ID}/sclrepl"
ARG SCALLOPY_URL="https://scallop-lang.github.io/artifacts/scallopy/scallopy-${VERSION_ID}-cp39-cp39-manylinux_2_27_x86_64.whl"

# ----------------------------
# System setup
# ----------------------------
RUN apt-get update && apt-get -y upgrade && \
    apt-get -y install wget curl bzip2 && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash scallop_user
USER scallop_user
WORKDIR /home/scallop_user

ENV PATH="/home/scallop_user/.local/bin:${PATH}"

# ----------------------------
# Install Miniconda
# ----------------------------
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/scallop_user/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/home/scallop_user/miniconda3/bin:${PATH}"

# ----------------------------
# Install Scallop binaries
# ----------------------------
RUN mkdir -p /home/scallop_user/packages/scallop/bin \
             /home/scallop_user/packages/scallop/lib \
             /home/scallop_user/labs/lab1 \
             /home/scallop_user/labs/lab2

WORKDIR /home/scallop_user/packages/scallop/bin

RUN wget ${SCLI_URL} && \
    wget ${SCLREPL_URL} && \
    chmod +x scli sclrepl

ENV PATH="/home/scallop_user/packages/scallop/bin:${PATH}"

# ----------------------------
# Create conda environment
# ----------------------------
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ----------------------------
# Install mamba (faster and more reliable than conda)
# ----------------------------
RUN conda install -n base -c conda-forge mamba -y

RUN conda create -n scallop-env python=3.9 -y

# ----------------------------
# Configure shell to auto-activate env
# ----------------------------
RUN echo ". /home/scallop_user/miniconda3/etc/profile.d/conda.sh" >> /home/scallop_user/.bashrc && \
    echo "conda activate scallop-env" >> /home/scallop_user/.bashrc

# ----------------------------
# Use conda environment for subsequent RUN commands
# ----------------------------
SHELL ["conda", "run", "-n", "scallop-env", "/bin/bash", "-c"]

# ----------------------------
# Install PyTorch (using mamba for better reliability)
# ----------------------------
# RUN pip install torch torchvision torchaudio \
#     --index-url https://download.pytorch.org/whl/cu121 && \
#     pip install transformers==4.46.3

# ----------------------------
# Install scallopy + Python libraries
# ----------------------------
WORKDIR /home/scallop_user/packages/scallop/lib

RUN wget ${SCALLOPY_URL}

RUN pip install --upgrade pip && \
    pip install $(basename ${SCALLOPY_URL}) && \
    pip install notebook ipywidgets tqdm matplotlib scikit-learn pandas seaborn datasets

# ----------------------------
# Copy project NeuroSymbolic
# ----------------------------
RUN mkdir -p /home/scallop_user/labs/dgl && \
    chown -R scallop_user:scallop_user /home/scallop_user/labs

WORKDIR /home/scallop_user/labs/dgl

COPY --chown=scallop_user:scallop_user . .

WORKDIR /home/scallop_user/labs/dgl

# ----------------------------
# Install packages
# ----------------------------
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------
# Keep container alive
# ----------------------------
CMD ["sleep", "infinity"]
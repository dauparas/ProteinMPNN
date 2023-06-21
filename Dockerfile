# Original Copyright 2021 DeepMind Technologies Limited
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

FROM public.ecr.aws/amazonlinux/amazonlinux:2 as base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV NVARCH x86_64
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.7 brand=tesla,driver>=450,driver<451 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 brand=tesla,driver>=510,driver<511 brand=unknown,driver>=510,driver<511 brand=nvidia,driver>=510,driver<511 brand=nvidiartx,driver>=510,driver<511 brand=geforce,driver>=510,driver<511 brand=geforcertx,driver>=510,driver<511 brand=quadro,driver>=510,driver<511 brand=quadrortx,driver>=510,driver<511 brand=titan,driver>=510,driver<511 brand=titanrtx,driver>=510,driver<511"
ENV NV_CUDA_CUDART_VERSION 11.7.99-1
ENV CUDA_VERSION 11.7.1
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NV_LIBNCCL_PACKAGE_NAME libnccl
ENV NVIDIA_PRODUCT_NAME="CUDA"
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
ENV CUDA_HOME=/usr/local/cuda-11.7
LABEL com.nvidia.cudnn.version="8.5.0.96-1"

COPY cuda.repo-x86_64 /etc/yum.repos.d/cuda.repo

RUN NVIDIA_GPGKEY_SUM="d0664fbbdb8c32356d45de36c5984617217b2d0bef41b93ccecd326ba3b80c87" \
  && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel7/${NVARCH}/D42D0685.pub | sed '/^Version/d' > /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA \
  && echo "$NVIDIA_GPGKEY_SUM  /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA" | sha256sum -c --strict - \
  && yum install -y \
  cuda-cudart-11-7-11.7.99-1 \
  cuda-libraries-11-7-11.7.1-1 \
  cuda-nvtx-11-7-11.7.91-1 \
  libnpp-11-7-11.7.4.75-1 \
  libcublas-11-7-11.10.3.66-1 \
  libnccl-2.13.4-1+cuda11.7 \
  libcudnn8-8.5.0.96-1.cuda11.7 \
  wget-1.14 \
  git-2.39.2-1.amzn2.0.1  \
  unzip-6.0-57.amzn2.0.1 \
  which-2.20-7.amzn2.0.2 \
  && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
  && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf \
  && curl -L -O "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" \
  && unzip -qq awscli-exe-linux-$(uname -m).zip \
  && bash aws/install \
  && rm awscli-exe-linux-$(uname -m).zip \
  && amazon-linux-extras install python3.8 \
  && git clone https://github.com/dauparas/ProteinMPNN.git /opt/proteinmpnn && cd /opt/proteinmpnn && git checkout be1d37b6699dcd2283ab5b6fc8cc88774e2c80e9 \
  && rm -rf colab_notebooks examples inputs outputs training .git \
  && python3.8 -m venv /opt/venv \
  && source /opt/venv/bin/activate \
  && python3.8 -m ensurepip --upgrade \
  && python -m pip install -q --no-cache-dir \
  torch \
  numpy \
  && yum clean all \
  && rm -rf /var/cache/yum \
  && mkdir --parents /root/.cache/torch/hub/checkpoints

ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /opt/proteinmpnn

COPY run.sh run.sh
COPY protein_mpnn_run.py /opt/proteinmpnn/

ENTRYPOINT ["bash", "run.sh"]

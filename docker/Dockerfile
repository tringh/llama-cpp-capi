# =================================================================================================
# STAGE 1: Builder
#
# In this stage, we download and prepare CLion and C/C++ development tools.
# This keeps the final production image lean by not including temporary build tools.
# =================================================================================================
FROM ubuntu:24.04 AS builder

# Set versions using ARGs for easy updates
ARG CLION_VERSION=2025.2.4
ARG CMAKE_VERSION=3.31.2
ARG NINJA_VERSION=1.12.1
ARG CUDA_VERSION=13.0.2

ENV CLION_VERSION=${CLION_VERSION}
ENV CMAKE_VERSION=${CMAKE_VERSION}
ENV NINJA_VERSION=${NINJA_VERSION}
ENV CUDA_VERSION=${CUDA_VERSION}

# Install build-time dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    tar \
    unzip \
    wget \
    gnupg2 && \
    rm -rf /var/lib/apt/lists/*

# --- Download and Extract CLion ---
ARG CLION_INSTALL_DIR=/opt/jetbrains/clion
RUN mkdir -p ${CLION_INSTALL_DIR}
RUN curl -L "https://download.jetbrains.com/cpp/CLion-${CLION_VERSION}.tar.gz" | \
    tar -xz -C ${CLION_INSTALL_DIR} --strip-components=1

# --- Download and Extract CMake ---
ARG CMAKE_INSTALL_DIR=/opt/cmake
RUN mkdir -p ${CMAKE_INSTALL_DIR} && \
    curl -L "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" | \
    tar -xz -C ${CMAKE_INSTALL_DIR} --strip-components=1

# --- Download and Extract Ninja ---
ARG NINJA_INSTALL_DIR=/opt/ninja
RUN mkdir -p ${NINJA_INSTALL_DIR} && \
    curl -L "https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/ninja-linux.zip" -o /tmp/ninja.zip && \
    unzip -q /tmp/ninja.zip -d ${NINJA_INSTALL_DIR} && \
    rm /tmp/ninja.zip

# =================================================================================================
# STAGE 2: Final Production Image
#
# This is the final, optimized image. It includes C/C++ toolchains, CUDA toolkit,
# development tools, and the CLion server, ready for remote development.
# =================================================================================================
FROM ubuntu:24.04

# Set versions again for use in this stage
ARG CLION_VERSION=2025.2.1
ARG CUDA_VERSION=13.0.2
ENV CLION_VERSION=${CLION_VERSION}
ENV CUDA_VERSION=${CUDA_VERSION}

# --- Environment Setup ---
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# --- Install System Dependencies & Essential Dev Tools ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # SSH Server
    openssh-server \
    ca-certificates \
    # Version Control & Utilities
    git \
    wget \
    curl \
    unzip \
    nano \
    sudo \
    # C/C++ Build Tools & Compilers
    build-essential \
    gcc \
    g++ \
    gdb \
    clang \
    clang-format \
    clang-tidy \
    lldb \
    # Development Libraries
    libc6-dev \
    libstdc++-14-dev \
    libcurl4-openssl-dev \
    # <--- THIS IS THE CORRECTED LINE
    # Memory & Performance Analysis Tools
    valgrind \
    # Additional Utilities
    pkg-config \
    autoconf \
    automake \
    libtool \
    make \
    # System monitoring
    procps \
    htop \
    # Documentation & Help
    man-db \
    manpages-dev \
    # Required for CUDA installation
    gnupg2 \
    software-properties-common && \
    rm -rf /var/lib/apt/lists/*

# --- Install NVIDIA CUDA Toolkit ---
# Add NVIDIA package repositories
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb

# Install CUDA Toolkit (without driver as we'll use host driver)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cuda-toolkit-13-0 \
    cuda-cudart-dev-13-0 \
    cuda-nvcc-13-0 \
    cuda-libraries-dev-13-0 \
    libcublas-dev-13-0 \
    libcurand-dev-13-0 && \
    rm -rf /var/lib/apt/lists/*

# --- Set up CUDA environment variables ---
ENV CUDA_HOME=/usr/local/cuda-13.0
ENV CUDA_PATH=/usr/local/cuda-13.0
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# --- Install CMake ---
ARG CMAKE_INSTALL_DIR=/opt/cmake
COPY --from=builder ${CMAKE_INSTALL_DIR} ${CMAKE_INSTALL_DIR}
ENV CMAKE_HOME=${CMAKE_INSTALL_DIR}

# --- Install Ninja ---
ARG NINJA_INSTALL_DIR=/opt/ninja
COPY --from=builder ${NINJA_INSTALL_DIR} ${NINJA_INSTALL_DIR}
ENV NINJA_HOME=${NINJA_INSTALL_DIR}

# --- Configure Environment PATH for all command-line tools ---

# 1. Set the PATH for non-login shells (e.g., `docker exec`)
ENV PATH="${CUDA_HOME}/bin:${CMAKE_HOME}/bin:${NINJA_HOME}:${PATH}"

# 2. For SSH login shells, write the DIRECT paths to the system-wide profile
RUN echo 'export PATH="/usr/local/cuda-13.0/bin:/opt/cmake/bin:/opt/ninja:${PATH}"' >> /etc/profile && \
    echo 'export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}"' >> /etc/profile && \
    echo 'export CUDA_HOME=/usr/local/cuda-13.0' >> /etc/profile

# --- Remove the conflicting default 'ubuntu' user ---
# This user often has UID 1000, which can conflict with the host user.
RUN if id "ubuntu" &>/dev/null; then userdel -r ubuntu; fi

# --- Create a non-root user 'developer' ---
RUN groupadd developer && \
    useradd --gid developer -m -s /bin/bash developer && \
    echo 'developer:password' | chpasswd && \
    usermod -aG sudo developer

# Allow passwordless sudo for convenience
RUN echo '%developer ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# --- Configure and Prepare SSH Server ---
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    mkdir -p /var/run/sshd && \
    ssh-keygen -A

# --- Create Project Workspace inside the user's home directory ---
RUN mkdir -p /home/developer/workspace && \
    chown developer:developer /home/developer/workspace

# --- Create .hushlogin to disable MOTD on login ---
RUN touch /home/developer/.hushlogin && \
    chown developer:developer /home/developer/.hushlogin

# --- Install CLion ---
ARG CLION_INSTALL_DIR=/opt/jetbrains/clion
COPY --from=builder ${CLION_INSTALL_DIR} ${CLION_INSTALL_DIR}

# --- Switch to developer user to register the IDE ---
USER developer
WORKDIR /home/developer

# [FIX] Prepend the export to .bashrc (Top of file strategy) and append to .profile
# This ensures CLion non-interactive shells still see the socket.
RUN echo "export SSH_AUTH_SOCK=/ssh-agent" > ~/.bashrc_temp && \
    ([ -f ~/.bashrc ] && cat ~/.bashrc >> ~/.bashrc_temp || true) && \
    mv ~/.bashrc_temp ~/.bashrc && \
    echo "export SSH_AUTH_SOCK=/ssh-agent" >> ~/.profile && \
    touch /home/developer/.hushlogin && \
    ${CLION_INSTALL_DIR}/bin/remote-dev-server.sh registerBackendLocationForGateway

# Switch back to root for final setup
USER root

# --- Setup Runtime Entrypoint for Permission Handling ---
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Expose the SSH port
EXPOSE 22

# Use the entrypoint to handle user permissions and then start the SSH server
ENTRYPOINT ["entrypoint.sh"]

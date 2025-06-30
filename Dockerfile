FROM nvcr.io/nvidia/pytorch:24.03-py3

# Define build arguments
ARG UID
ARG GID

# Ensure UID and GID are provided
RUN if [ -z "$UID" ] || [ -z "$GID" ]; then \
    echo "Error: UID and GID build arguments must be provided." >&2; \
    exit 1; \
    fi

# Set timezone and disable interactive prompts
ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

# 1) Install base packages (including keychain)

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        poppler-utils \
        git \
        sudo \
        vim \
        screen \
        software-properties-common \
        tzdata \
        fonts-noto \
        keychain && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*  # Cleanup

# 2) Set Python 3.11 as the default

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 && \
    update-alternatives --set python3 /usr/bin/python3.11

ENV PATH="/usr/bin:$PATH"

# Verify Python version as root (should print Python 3.11.x)
RUN python3 --version

# 3) Create a group and user

RUN groupadd --gid "${GID}" swaroopa_jinka && \
    useradd --uid "${UID}" --gid "${GID}" -m -s /bin/bash swaroopa_jinka && \
    echo "swaroopa_jinka:abcde1234" | chpasswd

# Switch to the new user
USER swaroopa_jinka

# Create and set the working directory
WORKDIR /home/swaroopa_jinka

# Verify Python version as the new user
RUN echo "Python version: $(python3 --version)"

# (Optional) Create or activate your Python virtual environment here
# Example:
# RUN python3 -m venv .venv && \
#     source .venv/bin/activate && \
#     pip install ...

CMD ["/bin/bash"]
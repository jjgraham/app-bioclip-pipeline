FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /workspace

# Install system dependencies for OpenCV core operations and RTSP stream decoding
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install modern SAM 3.1 dependencies, including the huggingface_hub tool
RUN pip install --no-cache-dir \
    flask \
    ultralytics \
    git+https://github.com/facebookresearch/sam3.git \
    pybioclip \
    pywaggle \
    timm \
    huggingface_hub

# Download the SAM 3.1 official weights locally using the correct Hugging Face repository
# (Saves to /workspace/sam3_b.safetensors)
RUN huggingface-cli download 1038lab/sam3 sam3.safetensors --local-dir /workspace --local-dir-use-symlinks False

COPY app.py /workspace/app.py

EXPOSE 5000
CMD ["python", "app.py"]

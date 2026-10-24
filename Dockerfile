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

# Install ML pipeline dependencies
RUN pip install --no-cache-dir \
    flask \
    ultralytics \
    git+https://github.com/facebookresearch/segment-anything.git \
    pybioclip \
    pywaggle \
    timm

# Pre-download official SAM ViT-H and YOLOv8x checkpoints into container layer
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O /workspace/sam_vit_h_4b8939.pth && \
    wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt -O /workspace/yolov8x.pt

COPY app.py /workspace/app.py

EXPOSE 5000
CMD ["python", "app.py"]

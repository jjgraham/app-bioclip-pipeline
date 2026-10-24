FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /workspace

# Install system dependencies for OpenCV core operations
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install modern orchestration libraries & pipelines
RUN pip install --no-cache-dir \
    flask \
    ultralytics \
    git+https://github.com \
    pybioclip \
    pywaggle

# Pull the dense SAM checkpoint directly into build caching
RUN wget https://fbaipublicfiles.com

COPY app.py /workspace/app.py

EXPOSE 5000
CMD ["python", "app.py"]

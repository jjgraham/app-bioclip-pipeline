FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /workspace

# Install system dependencies for OpenCV image processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install specific ML pipeline frameworks
RUN pip install --no-cache-dir \
    flask \
    ultralytics \
    git+https://github.com \
    pybioclip \
    pywaggle

# Pre-download weights inside the container layer to speed up node initialization
RUN wget https://fbaipublicfiles.com

COPY app.py /workspace/app.py

EXPOSE 5000
CMD ["python", "app.py"]

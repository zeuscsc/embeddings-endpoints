# Check for the correct tag for CUDA 11.8 on NVIDIA's Docker Hub. This is an example based on Ubuntu 20.04
# FROM nvidia/cuda:11.8.0-base-ubuntu20.04
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime


# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Update and install python and pip
RUN apt-get update && apt-get install -y python3-pip python3-dev && ln -sf python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# The work directory inside your container
WORKDIR /app

# Copy the contents from your host to your current location.
COPY . /app

# Install Python dependencies from requirements
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt

RUN python cache_model.py

# Set the environment to interactive again
ENV DEBIAN_FRONTEND=teletype

# Expose the port the app runs on
EXPOSE 8240

# Run the command to start uWSGI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8240"]
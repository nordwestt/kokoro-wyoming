FROM python:3.12-slim
ENV PYTHONPATH=/app
WORKDIR /app/src

# Download required model files
RUN mkdir -p /app/src
ADD --checksum=sha256:7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5 https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx /app/src/kokoro-v1.0.onnx
ADD --checksum=sha256:bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin /app/src/voices-v1.0.bin

# Install required packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Installing OpenVINO at the same time as main requirements doesn't work
COPY requirements-openvino.txt .
RUN pip install -r requirements-openvino.txt

COPY src/ /app/src/

ENTRYPOINT ["/usr/local/bin/python3", "main.py"]

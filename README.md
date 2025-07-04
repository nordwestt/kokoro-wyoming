# Kokoro Wyoming

This a [wyoming protocol](https://github.com/rhasspy/wyoming) server that implements [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) using the [Kokoro onnx](https://github.com/thewh1teagle/kokoro-onnx) python library.

Recent significant improvements from relvacode:

  - Updated to latest Kokoro model
  - Available voices are read directly from the model instead of hardcoding
  - Support for `SIGINT` and `SIGTERM` termination
  - Support for GPU accelerated inference (see [./docker-compose.intel.yml]())
  - Kokoro debug logging

Docker image avaiable at:
https://hub.docker.com/r/nordwestt/kokoro-wyoming 

Latest improvements available at:
```
docker pull ghcr.io/relvacode/kokoro-wyoming:latest
```

## Setup

Use one of the provided docker compose files in this repository (use Intel if you have a recent Intel iGPU).

Then go to Home Assistant -> Device & Services -> Add Integration -> Wyoming Protocol. Then, punch in the IP address of where the container is running, and the port (10210) by default.

Once setup, you can use `Kokoro` as an assistant TTS provider by going to Home Assistant -> Voice assistants and either edit or create an assistant, then select `Kokoro` as the Text-to-speech provider.

Options for available voices can be found in the [model voices documentation](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md).

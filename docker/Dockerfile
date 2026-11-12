From pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
RUN pip install --no-cache-dir onnx onnxruntime-gpu==1.20.2
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH}
WORKDIR /lab

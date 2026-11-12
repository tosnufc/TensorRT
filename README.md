# TensorRT lab: ResNet-50 from PyTorch to ONNX to TensorRT

A hands-on study of NVIDIA inference technologies on a single workstation.
We take one image classifier (**ResNet-50**) and run it in different runtimes and precisions,
measuring **accuracy**, **latency** and **throughput** for each:

```
PyTorch model ──export──> resnet50.onnx ──┬──> ONNX Runtime  (runs ONNX directly)
                                          └──> TensorRT      (compiles to a .plan engine: FP32 / FP16 / INT8)
```

The goal is to understand the trade-offs an architect has to weigh: speed vs. accuracy,
portability vs. performance, and build and operations cost.

## Hardware and software
| Item | Version |
|---|---|
| GPU | NVIDIA RTX 3080 Laptop, 16 GB (Ampere, sm_86: FP16/INT8 Tensor Cores, no FP8) |
| Driver / CUDA (host) | 595.99.02 / 13.2 |
| Runtime | Docker + NVIDIA Container Toolkit (`--gpus all`) |
| Base image | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` |
| TensorRT (planned) | NGC `nvcr.io/nvidia/tensorrt:26.10-py3` |

## Model and data
- **Model:** torchvision `resnet50`, weights `IMAGENET1K_V2` (1000 ImageNet classes).
- **Data:** [Imagenette](https://github.com/fastai/imagenette) `imagenette2-320`: 10 ImageNet classes,
  **3,925 validation images**. INT8 calibration uses **train** images only.
- **Label mapping:** Imagenette folder names (WNIDs) mapped to ImageNet class numbers:
  `n01440764:0, n02102040:217, n02979186:482, n03000684:491, n03028079:497,
  n03394916:566, n03417042:569, n03425413:571, n03445777:574, n03888257:701`

## Folder layout
```
docker/Dockerfile   # image "trt-lab-torch": PyTorch + ONNX + ONNX Runtime GPU
scripts/            # numbered steps, run in order
data/               # Imagenette (not in git, download it, see below)
models/             # resnet50.onnx and TensorRT engines (not in git, generated)
```

## Scripts
| Script | What it does |
|---|---|
| `01_export.py` | Exports ResNet-50 to `models/resnet50.onnx` (opset 17, dynamic batch, input `input`, output `logits`) |
| `02_inspect_onnx.py` | Lists the ONNX graph: 122 nodes (BatchNorm already folded into Conv) |
| `03_predict_one.py` / `03_predict_one_GPU.py` | Classifies a single image on CPU / GPU (golf ball = class 574) |
| `04a_check_mapping.py` | Checks the WNID-to-ImageNet label mapping |
| `04_accuracy_torch.py` | Accuracy on the val set, PyTorch on the GPU |
| `05_accuracy_ort.py` | Same accuracy check with ONNX Runtime (CUDA provider) |
| `06_speed_torch.py` | Speed test: batch 1/8/32, 20 warm-up and 100 timed runs, input already on the GPU |
| `07_speed_ort.py` | Same speed test with ONNX Runtime, input on the CPU (includes copies) |

## How to reproduce
```bash
# 1. Get the data
mkdir -p data && cd data
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar xzf imagenette2-320.tgz && cd ..

# 2. Build the image
docker build -t trt-lab-torch docker/

# 3. Run any script (example: export the ONNX model)
docker run --rm --gpus all -v $(pwd):/lab -v torch-cache:/root/.cache/torch \
  trt-lab-torch python scripts/01_export.py
```
- `-v $(pwd):/lab` mounts this folder into the container (the image's `WORKDIR` is `/lab`).
- `-v torch-cache:...` is a named volume, so PyTorch weights are downloaded only once.
- Files the container creates are owned by `root` (for example `models/resnet50.onnx`).

## Results so far
| Variant | Accuracy | Batch 1 latency | Batch 8 images/s | Batch 32 images/s | Size |
|---|---|---|---|---|---|
| PyTorch FP32 | 87.21% | 3.65 ms | 672 | 776 | 98 MB `.pth` |
| ONNX Runtime GPU | 87.18% | 2.46 ms | 639 | 727 | 102 MB `.onnx` |
| TensorRT FP32 | | | | | |
| TensorRT FP16 | | | | | |
| TensorRT INT8 | | | | | |

PyTorch is timed on the model only (input already on the GPU). ONNX Runtime is timed end to end
(CPU-to-GPU copy, model, GPU-to-CPU copy).

**Observations**
- Batch 1 is limited by **framework overhead**: ORT runs the whole graph in C++ and beats
  eager PyTorch even though it also copies the input.
- At batch 8 and 32 the GPU math dominates. Both use cuDNN, so speed is about equal, and ORT pays for the 19 MB copy.
- Accuracy matches across runtimes in FP32 (a 1-image difference is floating-point noise).

## Lessons learned (pitfalls)
1. **Pin ONNX Runtime to the image's CUDA version.** The newest `onnxruntime-gpu` (1.31) needs CUDA 13.
   This image has CUDA 12.4, so the Dockerfile pins `onnxruntime-gpu==1.20.2`.
2. **ORT falls back to CPU silently.** It does this both when a library fails to load and when `providers=` is misspelled.
   Always print `session.get_providers()`.
3. **cuDNN lives inside PyTorch's package folder** (`torch/lib`), so the Dockerfile adds it to
   `LD_LIBRARY_PATH`. Otherwise ORT fails with `libcudnn*.so.9: cannot open shared object file`.
4. **Benchmarks need warm-up and `torch.cuda.synchronize()`**, because GPU calls are asynchronous.
5. **TensorRT engines are tied to the GPU type and TensorRT version.** Build them where they will run,
   and never commit them.

## Roadmap
- [x] Baselines: PyTorch and ONNX Runtime (accuracy and speed)
- [ ] TensorRT with `trtexec`: FP32 and FP16 engines, layer fusion, per-layer profiling
- [ ] TensorRT accuracy check (Polygraphy and a Python inference script)
- [ ] INT8 with explicit quantization (NVIDIA Model Optimizer, Q/DQ nodes)
- [ ] Operations: optimization profiles, portability, VRAM use
- [ ] Next study steps: Triton Inference Server, TensorRT-LLM vs. vLLM, DCGM monitoring, Docker Compose

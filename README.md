# TensorRT lab: ResNet-50 from PyTorch to ONNX to TensorRT

A hands-on study of NVIDIA inference technologies on a single workstation.
We take one image classifier (**ResNet-50**) and run it in different runtimes and precisions,
measuring **accuracy**, **latency** and **throughput** for each:

```
PyTorch model ──export──> resnet50.onnx ──┬──> ONNX Runtime  (runs ONNX directly)
                                          ├──> trtexec ──> r50_fp32.plan
                                          ├──> ModelOpt AutoCast ──> resnet50_fp16.onnx ──> trtexec ──> r50_fp16.plan
                                          └──> ModelOpt INT8 (calibration) ──> resnet50_int8.onnx ──> trtexec ──> r50_int8.plan
```

The goal is to understand the trade-offs an architect has to weigh: speed vs. accuracy,
portability vs. performance, and build and operations cost.

## Hardware and software
| Item | Version |
|---|---|
| GPU | NVIDIA RTX 3080 Laptop, 16 GB (Ampere, sm_86: FP16/INT8 Tensor Cores, no FP8) |
| Driver / CUDA (host) | 595.99.02 / 13.2 |
| Runtime | Docker + NVIDIA Container Toolkit (`--gpus all`) |
| PyTorch image | `trt-lab-torch` (from `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`) |
| TensorRT image | NGC `nvcr.io/nvidia/tensorrt:26.10-py3`: **TensorRT 11.4.0**, CUDA 13.4, Python 3.12, Polygraphy 0.53.6 |
| Quantization | `nvidia-modelopt[onnx]` (installed with pip inside the TensorRT container) |

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
models/             # ONNX models, TensorRT engines, .npy data (not in git, generated)
```

## Scripts
| Script | Image | What it does |
|---|---|---|
| `01_export.py` | torch | Exports ResNet-50 to `models/resnet50.onnx` (opset 17, dynamic batch, input `input`, output `logits`) |
| `02_inspect_onnx.py` | torch | Lists the ONNX graph: 122 nodes (BatchNorm already folded into Conv) |
| `03_predict_one.py` / `03_predict_one_GPU.py` | torch | Classifies a single image on CPU / GPU (golf ball = class 574) |
| `04a_check_mapping.py` | torch | Checks the WNID-to-ImageNet label mapping |
| `04_accuracy_torch.py` | torch | Accuracy on the val set, PyTorch on the GPU |
| `05_accuracy_ort.py` | torch | Same accuracy check with ONNX Runtime (CUDA provider) |
| `06_speed_torch.py` | torch | Speed test: batch 1/8/32, 20 warm-up and 100 timed runs, input already on the GPU |
| `07_speed_ort.py` | torch | Same speed test with ONNX Runtime, input on the CPU (includes copies) |
| `08_make_calib.py` | torch | 500 random **train** images (seed 0) -> `models/calib.npy` for INT8 calibration |
| `09_prep_val.py` | torch | Preprocesses all val images once -> `models/val_images.npy` + `val_labels.npy` (ImageNet labels) |
| `10_accuracy_trt.py` | TensorRT | Accuracy of any engine: `python 10_accuracy_trt.py r50_fp16.plan` (Polygraphy `TrtRunner`, numpy only) |

Preprocessing (torchvision) and TensorRT inference run in **separate images** on purpose:
installing torch into the TensorRT 26.10 image would downgrade its CUDA 13.4 pip packages
(torch 2.15.0 pins `cuda-toolkit==13.2.2`). It also mirrors production, where preprocessing is a separate stage.

## How to reproduce
```bash
# 1. Get the data
mkdir -p data && cd data
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar xzf imagenette2-320.tgz && cd ..

# 2. Build the PyTorch image and pull the TensorRT image
docker build -t trt-lab-torch docker/
docker pull nvcr.io/nvidia/tensorrt:26.10-py3

# 3. PyTorch-side scripts (export, baselines, calibration and val data)
docker run --rm --gpus all -v $(pwd):/lab -v torch-cache:/root/.cache/torch \
  trt-lab-torch python scripts/01_export.py
docker run --rm -v $(pwd):/lab trt-lab-torch python scripts/08_make_calib.py
docker run --rm -v $(pwd):/lab trt-lab-torch python scripts/09_prep_val.py

# 4. TensorRT side: interactive container
docker run --rm -it --gpus all -v $(pwd):/lab -w /lab/models nvcr.io/nvidia/tensorrt:26.10-py3 bash
```
Inside the TensorRT container (`/lab/models`):
```bash
SHAPES="--minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 --maxShapes=input:32x3x224x224"

# FP32 engine
trtexec --onnx=resnet50.onnx --saveEngine=r50_fp32.plan $SHAPES

# FP16: convert the model first (TensorRT 11 has no --fp16 flag)
pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com "nvidia-modelopt[onnx]"
python -m modelopt.onnx.autocast --onnx_path resnet50.onnx --output_path resnet50_fp16.onnx
trtexec --onnx=resnet50_fp16.onnx --saveEngine=r50_fp16.plan $SHAPES

# INT8: explicit quantization with calibration data (TensorRT 11 has no --int8 flag)
python -m modelopt.onnx.quantization --onnx_path resnet50.onnx --quantize_mode int8 \
  --calibration_data_path calib.npy --output_path resnet50_int8.onnx
trtexec --onnx=resnet50_int8.onnx --saveEngine=r50_int8.plan $SHAPES

# Benchmark an engine at batch 1/8/32 (images/s = qps x batch)
for b in 1 8 32; do echo "== batch $b"; trtexec --loadEngine=r50_fp16.plan --shapes=input:${b}x3x224x224 2>&1 | grep -E "Throughput|Latency: min"; done
```
Accuracy of an engine (from the host):
```bash
docker run --rm --gpus all -v $(pwd):/lab nvcr.io/nvidia/tensorrt:26.10-py3 \
  python /lab/scripts/10_accuracy_trt.py r50_int8.plan
```

## Results
| Variant | Accuracy | Batch 1 latency | Batch 8 images/s | Batch 32 images/s | Size | Engine input |
|---|---|---|---|---|---|---|
| PyTorch FP32 | 87.21% | 3.65 ms | 672 | 776 | 98 MB `.pth` | fp32 |
| ONNX Runtime GPU | 87.18% | 2.46 ms | 639 | 727 | 102 MB `.onnx` | fp32 |
| TensorRT FP32 | 87.13% | 2.13 ms | 1,174 | 1,300 | 128 MB `.plan` | fp32 |
| TensorRT FP16 | 87.21% | 0.80 ms | 3,418 | 3,844 | 52 MB `.plan` | **fp16** |
| TensorRT INT8 | 86.98% | 0.45 ms | 7,200 | 8,341 | 29 MB `.plan` | fp32 |

- PyTorch and TensorRT are timed on the GPU only (TensorRT 11 `trtexec` excludes copies by default;
  `--includeDataTransfers` adds them). ONNX Runtime is timed end to end (CPU-to-GPU copy, model, GPU-to-CPU copy).
- Accuracy: 3,925 Imagenette val images, same preprocessing for every variant.
- Laptop GPU clocks vary by about 3%, so differences below about 5% are noise.

**Observations**
- **Compiling alone** (TensorRT FP32 vs. PyTorch) gives about **1.7x** (layer fusion + kernels auto-tuned for this GPU).
- **FP16** gives another **~3x** on Tensor Cores with **no accuracy loss**: the safe default for CNNs.
- **INT8** doubles FP16 again (**~10.7x PyTorch** at batch 32) for **-0.23 points** accuracy.
  Whether that is acceptable is a business decision, which needs an accuracy gate.
- **Batch size is the capacity knob:** going from batch 8 to 32 adds only 10–15% throughput but costs 3.5x latency.
  The GPU is nearly saturated at batch 8.
- Batch 1 in PyTorch is limited by **framework overhead**. ORT and TensorRT run the graph without Python per-op dispatch.
- 1–9 image differences between FP32 variants are floating-point noise (different kernels, TF32 on Ampere).

## Architect summary
1. **Pipeline in TensorRT 11:** `ONNX -> (ModelOpt AutoCast | ModelOpt quantization) -> ONNX variant -> trtexec -> engine`.
   Precision is a **model artifact**, not a build flag. Version each variant and gate it on accuracy in CI.
2. **Engines are build artifacts**, tied to GPU type (sm_86) and TensorRT version (11.4.0). Build them in CI for each GPU SKU,
   keep the ONNX file as the portable source, and never commit engines.
3. **The input dtype is part of the API contract:** the FP16 engine expects fp16 input, while the FP32 and INT8 engines expect fp32.
   Clients and Triton configs must match (`data_type`).
4. **Version unit = NGC release (YY.MM).** TensorRT, CUDA and Triton come as a tested set. Upgrade them together and rebuild all engines.
5. **Sizing:** on this GPU, traffic that needs ~10 GPUs with PyTorch needs ~1 with TensorRT INT8.

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
6. **TensorRT 11 removed `--fp16`, `--int8`, `--best` and calibrators.** Networks are always strongly typed.
   Use ModelOpt AutoCast (FP16) and ModelOpt quantization (INT8 Q/DQ). Many tutorials and LLM answers still show TensorRT 10 flags.
7. **Engine input types can change per variant** (AutoCast made the input fp16). Read the dtype from the engine
   (`runner.get_input_metadata()`) instead of hardcoding it.
8. **The TensorRT 26.10 image runs CUDA 13.4 on a 13.2 host driver** through *CUDA forward compatibility*
   (the banner shows it). This is officially supported on data-center GPUs only. In production, keep drivers current or plan for it.
9. **Don't `pip install torch` into the TensorRT image** without checking: the torch wheel's pinned CUDA packages would
   downgrade the image's CUDA stack.
10. **ModelOpt re-saved the input `resnet50.onnx`** (its timestamp changed). Graph and accuracy were verified unchanged,
    but source artifacts should be read-only and checksummed (`sha256sum`).
11. `trtexec` benchmarks report **qps (batches per second)**: images/s = qps x batch.

## Roadmap
- [x] Baselines: PyTorch and ONNX Runtime (accuracy and speed)
- [x] TensorRT with `trtexec`: FP32 and FP16 engines
- [x] TensorRT accuracy check (Polygraphy `TrtRunner`)
- [x] INT8 with explicit quantization (NVIDIA Model Optimizer, Q/DQ nodes)
- [ ] Optional: layer-fusion view, per-layer profiling, `--optShapes` tuning, VRAM comparison
- [ ] Next study steps: **Triton Inference Server** (ONNX + FP16/INT8 engines, dynamic batching, `perf_analyzer`),
      TensorRT-LLM vs. vLLM, DCGM monitoring, Docker Compose

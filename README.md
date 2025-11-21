下面是整理好的 **中英文双语 README.md**，你可以直接覆盖现在的文件使用。

````markdown
# ComfyUI_RemoteQwenSuite / 远程 Qwen 套件

Offload Qwen-VL text/image encoding to a **remote GPU server** and keep your local
ComfyUI environment lightweight.

将 Qwen-VL 的文本 / 图像编码工作卸载到 **远程 GPU 服务器**，减轻本地 ComfyUI
环境的显存与内存压力，让本地机器更轻量。

This project has **two parts** / 本项目分为 **两部分**：

1. `comfy_node/` – custom nodes for ComfyUI (local side)  
   `comfy_node/` – 本地 ComfyUI 自定义节点  
2. `remote_server/` – FastAPI service running Qwen-VL (remote side)  
   `remote_server/` – 运行 Qwen-VL 的远程 FastAPI 服务

---

## 1. Features / 功能特性

- **Remote Qwen CLIP encoder（远程 Qwen CLIP 编码器）**
  - Use Qwen2.5-VL (or compatible models) as the text / image encoder.  
    使用 Qwen2.5-VL（或兼容模型）作为文本 / 图像编码器。
  - HTTP-based protocol (`/encode`) so the encoder can run on another GPU box.  
    通过 HTTP 协议（`/encode` 接口）对接，可运行在另一台 GPU 服务器上。

- **No-CLIP / Selective-CLIP checkpoint loaders（无 CLIP / 可选 CLIP 的模型加载）**
  - `CheckpointLoaderNoCLIP`: load **MODEL + VAE** only, skip CLIP completely.  
    `CheckpointLoaderNoCLIP`：仅加载 **模型 + VAE**，完全跳过本地 CLIP。
  - `CheckpointLoaderSelective`: boolean flag `load_clip` to decide whether to
    build a local CLIP.  
    `CheckpointLoaderSelective`：通过布尔开关 `load_clip` 控制是否加载本地 CLIP。

- **Reference-latent injection（参考图 latent 注入）**
  - `RemoteQwenImageEditPlusRef`: encode 1–4 reference images with VAE and attach
    them as `reference_latents` to an existing CONDITIONING, matching Qwen-Image-Edit
    / Rapid-AIO style workflows.  
    `RemoteQwenImageEditPlusRef`：使用 VAE 对 1–4 张参考图像进行编码，并以
    `reference_latents` 的形式挂载到已有 CONDITIONING 上，兼容 Qwen-Image-Edit /
    Rapid-AIO 风格的工作流。

---

## 2. Repository layout / 仓库结构

```text
ComfyUI_RemoteQwenSuite/
├── comfy_node/                # ComfyUI custom nodes (local) / 本地自定义节点
│   ├── __init__.py
│   ├── load_checkpoint_no_clip.py
│   ├── remote_checkpoint.py
│   ├── remote_clip.py
│   ├── remote_edit_ref.py
│
├── remote_server/             # Qwen-VL encoder service (remote) / 远程编码服务
│   ├── remote_text_encoder_server.py
│   ├── remote_text_encoder_server_NF4.py
│   ├── start_qwen_full.bat
│   ├── start_qwen_nf4.bat
│   └── requirements.txt
│
├── example/                   # example workflows (TBD) / 示例工作流（待补）
└── README.md
````

---

## 3. Install – local ComfyUI node

本地 ComfyUI 自定义节点安装

1. **Clone this repo / 克隆仓库**

   ```bash
   git clone https://github.com/<your_name>/ComfyUI_RemoteQwenSuite.git
   ```

2. **Copy or symlink `comfy_node` into your ComfyUI `custom_nodes` folder**
   将 `comfy_node` 下的内容复制或软链接到你的 ComfyUI `custom_nodes` 目录：

   ```text
   <ComfyUI root>/
     custom_nodes/
       ComfyUI_RemoteQwenSuite/   <- contents of comfy_node/
   ```

3. **Restart ComfyUI / 重启 ComfyUI**

After restart, you should see the following nodes in ComfyUI:
重启后，你将在 ComfyUI 中看到以下节点：

* **Remote Qwen CLIP (HTTP)** – internal ID `RemoteQwenCLIP`
* **Remote Qwen ImageEditPlus Ref (1–4 imgs)** – `RemoteQwenImageEditPlusRef`
* **Load Checkpoint (No CLIP)** – `CheckpointLoaderNoCLIP`
* **Load Checkpoint (Selective CLIP)** – `CheckpointLoaderSelective`

---

## 4. Install – remote Qwen server

远程 Qwen 服务器安装

On your remote GPU server:
在远程 GPU 服务器上：

1. **Copy `remote_server/` to the machine**
   将 `remote_server/` 目录复制到远程机器，例如：

   ```text
   D:\Models\remote_qwen_server\
   ```

2. **Create a Python environment and install dependencies**
   创建 Python 环境（conda / venv 均可），并安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

   Minimal dependencies / 最小依赖包括：

   * `torch` (CUDA / ROCm / XPU build as you like)
   * `transformers`
   * `accelerate`
   * `fastapi`
   * `uvicorn[standard]`
   * `qwen-vl-utils`
   * `bitsandbytes` (if you use the NF4 / 4bit path / 若使用 NF4 量化)

3. **Choose one of the server scripts / 选择要启动的服务脚本**

   * `remote_text_encoder_server.py`

     * Uses 4bit NF4 quantization on CUDA via `BitsAndBytesConfig`.
       通过 `BitsAndBytesConfig` 在 CUDA 上使用 4bit NF4 量化。
   * `remote_text_encoder_server_NF4.py`

     * Full-precision (BF16/FP16) version.
       全精度（BF16/FP16）版本。

4. **Start the server / 启动服务示例**

   ```bash
   # full precision / 全精度
   python remote_text_encoder_server_NF4.py

   # 4bit NF4 (on CUDA) / NF4 量化（CUDA）
   python remote_text_encoder_server.py
   ```

   Or use the provided `start_qwen_full.bat` / `start_qwen_nf4.bat` on Windows.
   在 Windows 上也可以直接使用提供的 `start_qwen_full.bat` /
   `start_qwen_nf4.bat`。

The server will listen on `0.0.0.0:8008` and exposes:
服务默认监听 `0.0.0.0:8008`，并提供以下接口：

* `GET /health` – health check / 健康检查
* `POST /encode` – main encoding endpoint / 主编码接口

---

## 5. Node usage (high level)

节点使用说明（高层概览）

### 5.1 `RemoteQwenCLIP`

**Inputs / 输入**

* `server_url`: e.g. `http://<remote-ip>:8008`
* `mode`: `"txt"` or `"img"`

  * `"txt"` – text-only encoding / 仅文本编码
  * `"img"` – multi-modal encoding with image / 文本 + 图像多模态编码
* `text` / `negative_text` – positive & negative prompts / 正向 & 反向提示词
* `image` – optional; required when `mode="img"` / 可选，在多模态模式下需要传入参考图

**Outputs / 输出**

* `positive`: CONDITIONING (positive) / 正向 CONDITIONING
* `negative`: CONDITIONING (negative) / 反向 CONDITIONING

These conditioning outputs can be fed into:
这些 CONDITIONING 可以进一步传入：

* `RemoteQwenImageEditPlusRef`（用于挂载 `reference_latents`），然后
* 采样节点：`KSampler`、`KSamplerAdvanced` 等。

---

### 5.2 `RemoteQwenImageEditPlusRef`

**Inputs / 输入**

* `conditioning`: from `RemoteQwenCLIP`
  来自 `RemoteQwenCLIP` 的 CONDITIONING。
* `vae`: VAE model from your checkpoint loader
  来自模型加载节点的 VAE。
* `image1`–`image4`: reference images
  1–4 张参考图像。
* `target_size`: resize target before VAE encoding (default 896)
  在送入 VAE 之前统一缩放到的尺寸（默认 896）。

**Output / 输出**

* `conditioning` with `reference_latents` appended in its extra values.
  带有 `reference_latents` 附加信息的 CONDITIONING，可直接用于 Image-Edit / AIO
  类工作流。

---

## 6. TODO / 后续计划

* Add example ComfyUI workflows under `example/`.
  在 `example/` 目录下补充示例工作流。
* Add better one-click launcher scripts for both local and remote sides.
  为本地与远程分别提供更加完善的一键启动脚本。
* Support alternative models (e.g. Qwen3-VL) by making `MODEL_ID` configurable.
  通过配置 `MODEL_ID` 支持其他模型（例如 Qwen3-VL）。

```
```

### Docker 支持命令

#### NVIDIA：

```bash
docker build -f docker/Dockerfile.cuda -t qwen-remote:cuda .
docker run --gpus all -p 8008:8008 \
    -e QWEN_MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct" \
    qwen-remote:cuda
```

#### Intel XPU：

```bash
docker build -f docker/Dockerfile.xpu -t qwen-remote:xpu .
docker run --device /dev/dri -p 8008:8008 \
    -e QWEN_MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct" \
    qwen-remote:xpu
```

#### AMD ROCm：

```bash
docker build -f docker/Dockerfile.rocm -t qwen-remote:rocm .
docker run --device /dev/kfd --device /dev/dri -p 8008:8008 \
    -e QWEN_MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct" \
    qwen-remote:rocm
```

---


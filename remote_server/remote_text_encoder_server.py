import os
from typing import Optional, Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# 放在文件靠上的位置（import 后面都行）
QWEN_IMAGE_EDIT_SYSTEM_PROMPT = (
    "Describe key details of the input image (including any objects, characters, poses, "
    "facial features, clothing, setting, textures and style), then explain how the user's "
    "text instruction should alter, modify or recreate the image. Generate a new image "
    "that meets the user's requirements, which can vary from a small change to a completely "
    "new image using inputs as a guide."
)

# ==========================
# 配置区域：可以按需要改
# ==========================
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def pick_device() -> torch.device:
    """
    自动选一个设备：
    - 优先 CUDA（包括 ROCm 版的 torch，通常也是 'cuda'）
    - 其次 XPU（Intel）
    - 再其次 MPS（Apple）
    - 最后 CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()

# 根据设备简单选个 dtype
if DEVICE.type in ("cuda", "xpu", "mps"):
    DTYPE = torch.bfloat16
else:
    DTYPE = torch.float32


# ==========================
# FastAPI 初始化
# ==========================
app = FastAPI(title="Remote Qwen2.5-VL Text Encoder")

model: Optional[Qwen2_5_VLForConditionalGeneration] = None
processor: Optional[AutoProcessor] = None


class EncodeRequest(BaseModel):
    """
    通用请求：
    - mode="txt"：纯文本（T2I）
    - mode="img"：图 + 文（图生图 I2I）
    - prompt：正向提示词
    - negative_prompt：负向提示词（为空则用 ""）
    - image：当 mode="img" 时必填，可以是：
        - "file:///C:/xxx/xxx.png"
        - "http://... / https://..."
        - "data:image;base64,XXXX"
    """
    mode: Literal["txt", "img"] = "txt"
    prompt: str
    negative_prompt: Optional[str] = ""
    image: Optional[str] = None


class EncodeResponse(BaseModel):
    seq_len: int
    hidden_dim: int
    cond: list  # [[...], ...]  正向 hidden states
    uncond: list  # [[...], ...]  负向 hidden states


# ==========================
# 模型加载 / 编码核心逻辑
# ==========================

def load_model_and_processor():
    global model, processor

    if model is not None and processor is not None:
        return

    print(f"[RemoteTextEncoder] Loading Qwen2.5-VL model from {MODEL_ID} on {DEVICE} ...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map={"": DEVICE.type},  # 简单 map 到单设备
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()

    print("[RemoteTextEncoder] Model & processor loaded.")


def _build_messages_text_only(text: str):
    """
    纯文本模式（T2I）——对齐 TextEncodeQwenImageEdit 的行为：
    没有额外的 system 指令，只是单轮 user 文本。
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
            ],
        }
    ]


def _build_messages_image_text(image: str, text: str):
    """
    图 + 文模式（I2I / Qwen-Image-Edit-2509 风格）

    对齐 TextEncodeQwenImageEditPlus 的设计：
    - system：说明“先描述图，再按用户指令改图”
    - user：Picture 1: [image] + 用户 prompt
    """
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": QWEN_IMAGE_EDIT_SYSTEM_PROMPT},
            ],
        },
        {
            "role": "user",
            "content": [
                # 模拟 "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
                {"type": "text", "text": "Picture 1: "},
                {"type": "image", "image": image},
                # 接上你的文字指令
                {"type": "text", "text": "\n" + text},
            ],
        },
    ]


@torch.inference_mode()
def encode_single(text: str, image: Optional[str]) -> torch.Tensor:
    """
    对单个 (文本 / 图+文) 输入做一次前向，返回：
        hidden: (seq_len, hidden_dim) 的 float32 Tensor（已经在 CPU 上）

    这里使用 Qwen2.5-VL 的 ForConditionalGeneration，
    调用 forward(output_hidden_states=True)，
    取最后一层 hidden_states[-1] 作为“文本编码”。
    """
    assert model is not None and processor is not None

    if image is None:
        messages = _build_messages_text_only(text)
    else:
        messages = _build_messages_image_text(image, text)

    # chat template：把 messages 变成一串可 token 化的文本
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # 这里只要 encoder 部分，不需要 generation prompt
    )

    # 处理多模态部分（images/videos），支持 base64 / URL / file 等
    image_inputs, video_inputs = process_vision_info(messages)

    # 用 processor 统一打包
    # 重点：不 padding、不 truncation，避免任何自动长度裁剪
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        truncation=False,
        return_tensors="pt",
    )

    # 可选调试：打印 token 长度
    if os.getenv("QWEN_DEBUG_TOKENS", "0") == "1":
        input_ids = inputs.get("input_ids", None)
        if input_ids is not None:
            print(f"[RemoteTextEncoder] token length = {input_ids.shape[1]}")

    # 丢到模型设备上
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )

    # hidden_states：list[层数]，每个 (batch, seq, dim)
    last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
    last_hidden = last_hidden[0].detach().to("cpu")  # (seq_len, hidden_dim)

    # 为了下一步 JSON 化方便，统一转换为 float32
    if last_hidden.dtype != torch.float32:
        last_hidden = last_hidden.float()

    return last_hidden


# ==========================
# FastAPI 路由
# ==========================

@app.on_event("startup")
def startup_event():
    load_model_and_processor()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "model_id": MODEL_ID,
    }


def _pad_or_trim_to_len(t: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    把 (L, D) 的 tensor 调整到 (target_len, D)：
    - 如果 L > target_len：截断前面的多余部分
    - 如果 L < target_len：在末尾补 0
    """
    L, D = t.shape
    if L == target_len:
        return t
    if L > target_len:
        return t[:target_len, :]
    # L < target_len
    pad = torch.zeros(target_len - L, D, dtype=t.dtype)
    return torch.cat([t, pad], dim=0)


@app.post("/encode", response_model=EncodeResponse)
def encode(req: EncodeRequest):
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    if req.mode == "img" and not req.image:
        raise HTTPException(status_code=400, detail="mode='img' requires 'image' field.")

    # 正向编码
    cond_hidden = encode_single(
        text=req.prompt,
        image=(req.image if req.mode == "img" else None),
    )

    # 负向编码：如果 negative_prompt 为空，就用 ""（纯 uncond）
    neg_text = req.negative_prompt or ""
    uncond_hidden = encode_single(
        text=neg_text,
        image=(req.image if req.mode == "img" else None),
    )

    # 对齐长度：取最大长度，两边都 pad/裁剪到同一个 seq_len
    cond_len, hidden_dim = cond_hidden.shape
    uncond_len, hidden_dim2 = uncond_hidden.shape

    if hidden_dim != hidden_dim2:
        raise HTTPException(
            status_code=500,
            detail=f"hidden dim mismatch: {hidden_dim} vs {hidden_dim2}",
        )

    target_len = max(cond_len, uncond_len)
    cond_hidden = _pad_or_trim_to_len(cond_hidden, target_len)
    uncond_hidden = _pad_or_trim_to_len(uncond_hidden, target_len)

    seq_len = target_len

    return EncodeResponse(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        cond=cond_hidden.tolist(),
        uncond=uncond_hidden.tolist(),
    )


if __name__ == "__main__":
    import uvicorn

    # 本机所有网卡监听，方便局域网访问
    uvicorn.run(
        "remote_text_encoder_server:app",
        host="0.0.0.0",
        port=8008,
        reload=False,
    )

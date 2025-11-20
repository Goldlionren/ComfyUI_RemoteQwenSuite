########################################
# 1) RemoteQwenCLIP 节点（HTTP 远程编码）
########################################

# ComfyUI_Remote_QwenCLIP 自定义节点
#
# 功能：
# - 调用你的 remote_text_encoder_server.py 提供的 /encode 接口
# - 支持两种模式：
#     mode = "txt" : 纯文本（T2I）
#     mode = "img" : 图 + 文（图生图 I2I）
# - 输出：
#     positive: CONDITIONING（正向）
#     negative: CONDITIONING（负向）
#
# 使用方式：
# - server_url 一般填: http://<remote-ip>:8008
#   节点内部会自动在后面加 /encode
#
# - T2I：
#     mode = txt
#     text / negative_text 正常填写
#     image 输入可以留空
#
# - 图生图：
#     mode = txt
#     将原始图像（或者你希望喂给 Qwen-VL 的图像）接到 image 输入
#     同时填 text / negative_text
#
# - 输出的 positive / negative 可以直接接到 RemoteQwenImageEditPlusRef 节点 之后从RemoteQwenImageEditPlusRef 节点 连接 KSampler / KSamplerAdvanced 的
#   正/负 CONDITIONING 输入。

import base64
import io

import torch
import requests
import numpy as np
from PIL import Image

try:
    import comfy.model_management as model_management
except ImportError:
    model_management = None


class ComfyUI_Remote_QwenCLIP:
    """
    调用远程 Qwen2.5-VL Text/Image Encoder，生成正向 / 负向 CONDITIONING。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server_url": (
                    "STRING",
                    {
                        "default": "http://127.0.0.1:8008",
                        "multiline": False,
                    },
                ),
                "mode": (
                    "STRING",
                    {
                        "default": "txt",
                        "choices": ["txt", "img"],
                    },
                ),
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "negative_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
            },
            "optional": {
                # ComfyUI 中的 IMAGE: 形状 [B, H, W, C], 值范围 [0,1]
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "conditioning/remote"

    def _get_device(self):
        if model_management is not None:
            try:
                return model_management.get_torch_device()
            except Exception:
                pass

        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _image_to_base64_data_uri(self, image_tensor: torch.Tensor) -> str:
        """
        将 ComfyUI 的 IMAGE（[B, H, W, C]，0~1 float）转成 data:image/png;base64,... 字符串。
        只取 batch 中的第 0 张。
        """
        if image_tensor is None:
            raise ValueError("image tensor is None in _image_to_base64_data_uri")

        # 取 batch 中第 0 张，形状 [H, W, C]
        img = image_tensor[0].detach().clamp(0.0, 1.0).cpu().numpy()
        # 转成 [H, W, C] uint8
        img = (img * 255.0).round().astype(np.uint8)

        # 确保是 RGB
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        pil_img = Image.fromarray(img, mode="RGB")

        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return "data:image/png;base64," + b64

    def encode(self, server_url, mode, text, negative_text, image=None):
        device = self._get_device()

        server_url = server_url.rstrip("/")
        url = server_url + "/encode"

        if mode not in ("txt", "img"):
            raise ValueError(f"Unsupported mode: {mode}. Must be 'txt' or 'img'.")

        payload = {
            "mode": mode,
            "prompt": text,
            "negative_prompt": negative_text or "",
        }

        if mode == "img":
            if image is None:
                raise ValueError("mode='img' 需要提供 image 输入。")
            try:
                image_data_uri = self._image_to_base64_data_uri(image)
            except Exception as e:
                raise RuntimeError(f"将 IMAGE 转为 base64 失败: {e}")
            payload["image"] = image_data_uri

        try:
            resp = requests.post(url, json=payload, timeout=120)
        except Exception as e:
            raise RuntimeError(f"[ComfyUI_Remote_QwenCLIP] 请求远程服务器失败: {e}")

        if resp.status_code != 200:
            raise RuntimeError(
                f"[ComfyUI_Remote_QwenCLIP] 远程服务器返回错误状态码 {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        if "cond" not in data or "uncond" not in data:
            raise RuntimeError(
                f"[ComfyUI_Remote_QwenCLIP] 返回的 JSON 中缺少 cond/uncond 字段: {data}"
            )

        cond_list = data["cond"]
        uncond_list = data["uncond"]

        # cond_list / uncond_list: [seq_len, hidden_dim]
        # 转成 torch.Tensor，并加 batch 维度 -> [1, seq_len, hidden_dim]
        cond_tensor = torch.tensor(cond_list, dtype=torch.float32).unsqueeze(0)
        uncond_tensor = torch.tensor(uncond_list, dtype=torch.float32).unsqueeze(0)

        cond_tensor = cond_tensor.to(device)
        uncond_tensor = uncond_tensor.to(device)

        # 这里不额外放 pooled_output 等，先用简单结构。
        positive = [[cond_tensor, {}]]
        negative = [[uncond_tensor, {}]]

        return (positive, negative)

import math
import comfy.utils
import node_helpers

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



########################################
# 2) RemoteQwenImageEditPlusRef 节点（挂 reference_latents）
########################################

class RemoteQwenImageEditPlusRef:
    """
    把 1~4 张参考图编码成 reference_latents 挂到已有的 CONDITIONING 上，
    用于 Qwen-Image-Edit / Qwen-Rapid-AIO 这类模型的图生图。
    conditioning 一般来自你的 Remote Qwen CLIP (HTTP，mode=txt)。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "vae": ("VAE",),
                "image1": ("IMAGE",),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "target_size": (
                    "INT",
                    {
                        "default": 896,
                        "min": 128,
                        "max": 2048,
                        "step": 32,
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = "advanced/conditioning"

    @staticmethod
    def _encode_ref_latent(vae, image, target_size: int):
        """
        基本照抄 TextEncodeQwenImageEditPlus 的 VAE 缩放逻辑：
        - 把图片缩放到接近 target_size x target_size，总像素相近
        - 宽高对齐到 32 的倍数
        - 用 VAE.encode 得到 latent
        """
        if image is None:
            return None

        # image: [B, H, W, C] -> [B, C, H, W]
        samples = image.movedim(-1, 1)

        # 等比例缩放到 area 接近 target_size^2
        total_area = int(target_size * target_size)
        h, w = samples.shape[2], samples.shape[3]
        scale_by = math.sqrt(total_area / (w * h))

        new_h = max(32, int(h * scale_by / 32) * 32)
        new_w = max(32, int(w * scale_by / 32) * 32)

        # 上采样到 new_w x new_h
        up = comfy.utils.common_upscale(samples, new_w, new_h, "lanczos", "center")

        # 回到 [B, H, W, C] 再丢给 VAE（只用前 3 个通道）
        img_for_vae = up.movedim(1, -1)[:, :, :, :3]
        ref_latent = vae.encode(img_for_vae)
        return ref_latent

    def apply(
        self,
        conditioning,
        vae,
        image1,
        image2=None,
        image3=None,
        image4=None,
        target_size=896,
    ):
        images = [image1, image2, image3, image4]
        ref_latents = []

        for img in images:
            if img is not None:
                rl = self._encode_ref_latent(vae, img, target_size)
                if rl is not None:
                    ref_latents.append(rl)

        if ref_latents:
            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                {"reference_latents": ref_latents},
                append=True,
            )

        return (conditioning,)


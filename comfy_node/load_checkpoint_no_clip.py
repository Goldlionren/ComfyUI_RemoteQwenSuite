# load_checkpoint_no_clip.py
import comfy.sd
import folder_paths


class CheckpointLoaderNoCLIP:
    """
    只从 checkpoint 里加载:
      - MODEL (UNet 等)
      - VAE
    完全跳过 CLIP 的构建和加载。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 和官方一样，从 checkpoints 列表里选
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = ("MODEL", "VAE")
    RETURN_NAMES = ("model", "vae")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"   # 出现在 Loaders 分类里
    DESCRIPTION = "Load checkpoint without CLIP (model + VAE only)."

    def load_checkpoint(self, ckpt_name):
        # 拼完整路径
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        # 关键：output_clip=False
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=False,
            # 兼容老版本签名，多带这个参数没有问题
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

        # 兼容不同版本返回值长度（有的返回 3 个，有的返回 4 个）
        if not isinstance(out, tuple):
            raise RuntimeError("Unexpected return type from load_checkpoint_guess_config")

        if len(out) == 4:
            model, _clip, vae, _clipvision = out
        elif len(out) == 3:
            model, _clip, vae = out
        else:
            raise RuntimeError(
                f"Unexpected tuple length from load_checkpoint_guess_config: {len(out)}"
            )

        # 只把 model 和 vae 返回给前端
        return (model, vae)

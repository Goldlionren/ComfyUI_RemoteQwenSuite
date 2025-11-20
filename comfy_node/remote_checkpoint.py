# remote_checkpoint.py
import comfy.sd
import folder_paths


class CheckpointLoaderSelective:
    """
    从 checkpoint 里加载:
      - MODEL (UNet 等)
      - VAE
    并通过一个布尔开关决定是否真正加载 CLIP。

    load_clip = False 时:
      - 不构建 CLIP 模型, 不占显存
      - CLIP 输出为 None (不要往下游连就行)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "load_clip": (
                    "BOOLEAN",
                    {
                        "default": False,  # 你现在主要用远程 Qwen, 默认关掉本地 CLIP
                        "label_on": "Load CLIP",
                        "label_off": "Skip CLIP",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name, load_clip):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        # 关键点: 用 load_clip 控制 output_clip
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=load_clip,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

        if not isinstance(out, tuple):
            raise RuntimeError("Unexpected return type from load_checkpoint_guess_config")

        if len(out) == 4:
            model, clip, vae, _clipvision = out
        elif len(out) == 3:
            model, clip, vae = out
        else:
            raise RuntimeError(
                f"Unexpected tuple length from load_checkpoint_guess_config: {len(out)}"
            )

        # 如果 load_clip=False, 确保输出的是 None, 而且实际不会构建 CLIP
        if not load_clip:
            clip = None

        return (model, clip, vae)

# __init__.py  （放在 ComfyUI_RemoteQwenSuite 里）
from .remote_clip import ComfyUI_Remote_QwenCLIP
from .remote_edit_ref import RemoteQwenImageEditPlusRef

from .load_checkpoint_no_clip import CheckpointLoaderNoCLIP
from .remote_checkpoint import CheckpointLoaderSelective




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



########################################
# 3) 统一的映射字典
########################################

NODE_CLASS_MAPPINGS = {
    # key 是节点内部 ID，保持和你原来一样就不会炸旧工程
    "RemoteQwenCLIP": ComfyUI_Remote_QwenCLIP,
    "RemoteQwenImageEditPlusRef": RemoteQwenImageEditPlusRef,
    # 新增：
    "CheckpointLoaderNoCLIP": CheckpointLoaderNoCLIP,
    "CheckpointLoaderSelective": CheckpointLoaderSelective,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # 这里是左下角节点搜索列表里显示的名字
    "RemoteQwenCLIP": "Remote Qwen CLIP (HTTP)",
    "RemoteQwenImageEditPlusRef": "Remote Qwen ImageEditPlus Ref (1-4 imgs)",
    # 在 UI 里的显示名
    "CheckpointLoaderNoCLIP": "Load Checkpoint (No CLIP)",
    "CheckpointLoaderSelective": "Load Checkpoint (Selective CLIP)",
}

# 本地权重（不进 Git）

仓库里只有本说明。`lite_ddpm.pt`、`osediff.pkl`、`sd21-base/` 等大文件已被排除，克隆后在本机下载。

## Lite 基线

```powershell
.\.venv\Scripts\python.exe scripts/train_lite.py
```

会写出 `weights/lite_ddpm.pt`。不要用官方验证集 GT 训练后再在同一验证集上报分。

## OSEDiff

需要：

| 文件 | 来源 | 约大小 |
| --- | --- | --- |
| `osediff.pkl` | https://raw.githubusercontent.com/cswry/OSEDiff/main/preset/models/osediff.pkl | 20 MB |
| `sd21-base/` | HuggingFace `Manojb/stable-diffusion-2-1-base`（只要 fp16 safetensors） | ~2.5 GB |

一键（大文件失败时改用镜像直链，见 `scripts/_download_file.py`）：

```powershell
$env:HF_HUB_DISABLE_XET='1'
.\.venv\Scripts\python.exe scripts/_download_weights.py
```

UNet 若在 `huggingface.co` 超时，用：

```powershell
.\.venv\Scripts\python.exe scripts/_download_file.py `
  "weights/sd21-base/unet/diffusion_pytorch_model.fp16.safetensors" `
  "https://hf-mirror.com/Manojb/stable-diffusion-2-1-base/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors"
```

不要把 `sd21-base` 或官方赛题图提交进 Git。

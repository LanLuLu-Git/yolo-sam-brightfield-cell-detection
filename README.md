# YOLO + SAM 细胞检测与分割

基于 [ultralytics](https://github.com/ultralytics/ultralytics)（YOLO，已做改进）与 [segment-anything](https://github.com/facebookresearch/segment-anything)（SAM）的细胞图像检测与分割项目。

## 目录结构

```
├── XTCpredictV21.py          # 主预测脚本（YOLO + SAM 结合）
├── jpkautorun.py             # 批量自动运行脚本
├── XTCmoutiyobject.py        # SAM 多目标分割
├── export_onnx.py            # 导出 ONNX
├── 版本信息.py                # 打印库版本信息
├── ultralytics-main/         # 修改后的 ultralytics 源码（含 change_model/ 改进模块）
├── segment-anything-main/    # 仅保留 SAM 权重（源码走 pip 依赖）
├── requirements.txt          # pip 依赖
├── environment.yml           # conda 环境
└── .gitignore / README.md
```

## 环境安装

conda：

```bash
conda env create -f environment.yml
conda activate YOLOSAM
```

或 pip：

```bash
pip install -r requirements.txt
```

依赖中已包含 `ultralytics` 与 `segment-anything`。本项目对 ultralytics 源码做了改进（新增 `ultralytics/change_model/` 下的 FEB、PFAE、Di_SpAM、FRCAM、WTFD、Restormer、CAFM_184 等模块），请使用本仓库 `ultralytics-main/` 中的修改版而非 pip 默认版。

## 模型权重（未随仓库上传）

所有 `*.pt` / `*.pth` 权重已加入 `.gitignore`，需自行准备：

- **SAM 权重** `sam_vit_h_4b8939.pth`（约 2.4G）：
  ```bash
  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P segment-anything-main/models/
  ```
- **YOLO 官方预训练权重**（yolo11n/s/m.pt、yolov8n.pt）：由 `pip install ultralytics` 自动下载，或从 ultralytics 官方获取。
- **训练得到的权重**（`runs/**/weights/best.pt`、`ultralytics-main/weights/best.pt` 等）：为训练输出，需自行训练生成。

## 使用

修改脚本中的模型路径与输入图像路径后运行，例如：

```bash
python XTCpredictV21.py
```

> 注：脚本中部分路径为开发机上的绝对路径（如 `D:/Project/YOLOwithSAM/...`），在其他机器上运行时需按本机目录调整。

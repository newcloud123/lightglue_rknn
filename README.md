## LightGlue 模型部署流程说明



```bash
### 1. 生成 ONNX 模型
使用 `LightGlue-ONNX-main` 工程，通过 `dynamo.py` 导出 SuperPoint + LightGlue 的 ONNX 模型：
cd LightGlue-ONNX-main && \
python dynamo.py export superpoint \
  --num-keypoints 512 \
  -b 2 \
  -h 1024 \
  -w 1024 \
  -o weights/superpoint_lightglue_pipeline.onnx
### 2. ONNX 转 RKNN

进入模型转换目录，将生成的 ONNX 模型转换为 RK3588 平台可用的 RKNN 模型
cd lightglue_convert && \
python convert.py 512_50.onnx rk3588 fp

### 3.RK3588 上推理

在 RK3588 设备上编译并运行推理程序
cd lightglue_rknn_infer && \
mkdir build && \
cd build && \
cmake .. && \
make && \
./lightclue_infer ../512_50.rknn ./out.jpg
、、、
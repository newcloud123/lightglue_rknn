from pathlib import Path
from typing import Annotated, Optional

import cv2
import typer

from lightglue_dynamo.cli_utils import check_multiple_of
from lightglue_dynamo.config import Extractor, InferenceDevice

app = typer.Typer()


@app.callback()
def callback():
    """LightGlue Dynamo CLI"""


@app.command()
def export(
    extractor_type: Annotated[Extractor, typer.Argument()] = Extractor.superpoint,
    output: Annotated[
        Optional[Path],  # typer does not support Path | None # noqa: UP007
        typer.Option("-o", "--output", dir_okay=False, writable=True, help="Path to save exported model."),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "-b", "--batch-size", min=0, help="Batch size of exported ONNX model. Set to 0 to mark as dynamic."
        ),
    ] = 0,
    height: Annotated[
        int, typer.Option("-h", "--height", min=0, help="Height of input image. Set to 0 to mark as dynamic.")
    ] = 0,
    width: Annotated[
        int, typer.Option("-w", "--width", min=0, help="Width of input image. Set to 0 to mark as dynamic.")
    ] = 0,
    num_keypoints: Annotated[
        int, typer.Option(min=128, help="Number of keypoints outputted by feature extractor.")
    ] = 1024,
    fuse_multi_head_attention: Annotated[
        bool,
        typer.Option(
            "--fuse-multi-head-attention",
            help="Fuse multi-head attention subgraph into one optimized operation. (ONNX Runtime-only).",
        ),
    ] = False,
    opset: Annotated[int, typer.Option(min=16, max=20, help="ONNX opset version of exported model.")] = 17,
    fp16: Annotated[bool, typer.Option("--fp16", help="Whether to also convert to FP16.")] = False,
):
    """Export LightGlue to ONNX."""
    import onnx
    import torch
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
    from onnxruntime.transformers.float16 import convert_float_to_float16

    from lightglue_dynamo.models import DISK, LightGlue, Pipeline, SuperPoint,Pipeline2
    from lightglue_dynamo.ops import use_fused_multi_head_attention

    match extractor_type:
        case Extractor.superpoint:
            extractor = SuperPoint(num_keypoints=num_keypoints)
        case Extractor.disk:
            extractor = DISK(num_keypoints=num_keypoints)
    matcher = LightGlue(**extractor_type.lightglue_config)
    """ 
    checkpoints_path = "/data/luoshiyong/code/LightGlue-ONNX-main/gim_lightglue_100h.ckpt"
    state_dict = torch.load(checkpoints_path, map_location='cpu')
    if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('model.'):
            state_dict.pop(k)
        if k.startswith('superpoint.'):
            state_dict[k.replace('superpoint.', '', 1)] = state_dict.pop(k)
    extractor.load_state_dict(state_dict)

    state_dict = torch.load(checkpoints_path, map_location='cpu')
    if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('superpoint.'):
            state_dict.pop(k)
        if k.startswith('model.'):
            state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
    matcher.load_state_dict(state_dict)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
    pipeline = Pipeline(extractor, matcher).eval()

    if output is None:
        output = Path(f"weights/{extractor_type}_lightglue_pipeline.onnx")

    check_multiple_of(batch_size, 2)
    check_multiple_of(height, extractor_type.input_dim_divisor)
    check_multiple_of(width, extractor_type.input_dim_divisor)

    if height > 0 and width > 0 and num_keypoints > height * width:
        raise typer.BadParameter("num_keypoints cannot be greater than height * width.")

    if fuse_multi_head_attention:
        typer.echo(
            "Warning: Multi-head attention nodes will be fused. Exported model will only work with ONNX Runtime CPU & CUDA execution providers."
        )
        if torch.__version__ < "2.4":
            raise typer.Abort("Fused multi-head attention requires PyTorch 2.4 or later.")
        use_fused_multi_head_attention()

    dynamic_axes = {"images": {}, "keypoints": {}}
    if batch_size == 0:
        dynamic_axes["images"][0] = "batch_size"
        dynamic_axes["keypoints"][0] = "batch_size"
    if height == 0:
        dynamic_axes["images"][2] = "height"
    if width == 0:
        dynamic_axes["images"][3] = "width"
    dynamic_axes |= {"matches": {0: "num_matches"}, "mscores": {0: "num_matches"}}
    torch.onnx.export(
        pipeline,
        torch.zeros(batch_size or 2, extractor_type.input_channels, height or 256, width or 256),
        str(output),
        input_names=["images"],
        output_names=["keypoints", "matches", "mscores"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
    )
    onnx.checker.check_model(output)
    onnx.save_model(SymbolicShapeInference.infer_shapes(onnx.load_model(output), auto_merge=True), output)  # type: ignore
    typer.echo(f"Successfully exported model to {output}")
    if fp16:
        typer.echo(
            "Converting to FP16. Warning: This FP16 model should NOT be used for TensorRT. TRT provides its own fp16 option."
        )
        onnx.save_model(convert_float_to_float16(onnx.load_model(output)), output.with_suffix(".fp16.onnx"))


@app.command()
def infer(
    model_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to ONNX model.")],
    left_image_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to first image.")
    ],
    right_image_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to second image.")
    ],
    extractor_type: Annotated[Extractor, typer.Argument()] = Extractor.superpoint,
    output_path: Annotated[
        Optional[Path],  # noqa: UP007
        typer.Option(
            "-o",
            "--output",
            dir_okay=False,
            writable=True,
            help="Path to save output matches figure. If not given, show visualization.",
        ),
    ] = None,
    height: Annotated[
        int,
        typer.Option("-h", "--height", min=1, help="Height of input image at which to perform inference."),
    ] = 1024,
    width: Annotated[
        int,
        typer.Option("-w", "--width", min=1, help="Width of input image at which to perform inference."),
    ] = 1024,
    device: Annotated[
        InferenceDevice, typer.Option("-d", "--device", help="Device to run inference on.")
    ] = InferenceDevice.cpu,
    fp16: Annotated[bool, typer.Option("--fp16", help="Whether model uses FP16 precision.")] = False,
    profile: Annotated[bool, typer.Option("--profile", help="Whether to profile model execution.")] = False,
):
    """Run inference for LightGlue ONNX model."""
    import numpy as np
    import onnxruntime as ort

    from lightglue_dynamo import viz
    from lightglue_dynamo.preprocessors import DISKPreprocessor, SuperPointPreprocessor

    raw_images = [left_image_path, right_image_path]
    raw_images = [cv2.resize(cv2.imread(str(i)), (width, height)) for i in raw_images]
    images = np.stack(raw_images)
    match extractor_type:
        case Extractor.superpoint:
            images = SuperPointPreprocessor.preprocess(images)
        case Extractor.disk:
            images = DISKPreprocessor.preprocess(images)
    images = images.astype(np.float16 if fp16 and device != InferenceDevice.tensorrt else np.float32)

    session_options = ort.SessionOptions()
    session_options.enable_profiling = profile
    # session_options.optimized_model_filepath = "weights/ort_optimized.onnx"

    providers = [("CPUExecutionProvider", {})]
    if device == InferenceDevice.cuda:
        providers.insert(0, ("CUDAExecutionProvider", {}))
    elif device == InferenceDevice.tensorrt:
        providers.insert(0, ("CUDAExecutionProvider", {}))
        providers.insert(
            0,
            (
                "TensorrtExecutionProvider",
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "weights/.trtcache_engines",
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": "weights/.trtcache_timings",
                    "trt_fp16_enable": fp16,
                },
            ),
        )
    elif device == InferenceDevice.openvino:
        providers.insert(0, ("OpenVINOExecutionProvider", {}))

    session = ort.InferenceSession(model_path, session_options, providers)

    for _ in range(100 if profile else 1):
        keypoints, matches, mscores = session.run(None, {"images": images})
    if True:
         mask_np = matches[0].astype(bool)  # 取 batch 0 的掩码
         from collections import Counter
         print(Counter(mask_np.flatten()))
         idx0, idx1 = np.nonzero(mask_np)
         matches = np.stack([idx0, idx1], axis=1)  # (M, 2) 匹配坐标
         mscores = mscores[0][mask_np]  # 匹配分数
         nxx = np.zeros((matches.shape[0], 1), dtype=matches.dtype)
         matches = np.concatenate([nxx, matches], axis=1)   # [355, 3]
    print("keypoints shape = ",keypoints.shape)
    print("matches shape = ",matches.shape)
    print("mscores shape = ",mscores.shape)

    viz.plot_images(raw_images)
    viz.plot_matches(keypoints[0][matches[..., 1]], keypoints[1][matches[..., 2]], color="lime", lw=0.2)
    if output_path is None:
        viz.plt.show()
    else:
        viz.save_plot(output_path)


@app.command()
def trtexec(
    model_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to ONNX model or built TensorRT engine."),
    ],
    left_image_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to first image.")
    ],
    right_image_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to second image.")
    ],
    extractor_type: Annotated[Extractor, typer.Argument()] = Extractor.superpoint,
    output_path: Annotated[
        Optional[Path],  # noqa: UP007
        typer.Option(
            "-o",
            "--output",
            dir_okay=False,
            writable=True,
            help="Path to save output matches figure. If not given, show visualization.",
        ),
    ] = None,
    height: Annotated[
        int,
        typer.Option("-h", "--height", min=1, help="Height of input image at which to perform inference."),
    ] = 1024,
    width: Annotated[
        int,
        typer.Option("-w", "--width", min=1, help="Width of input image at which to perform inference."),
    ] = 1024,
    fp16: Annotated[bool, typer.Option("--fp16", help="Whether model uses FP16 precision.")] = False,
    profile: Annotated[bool, typer.Option("--profile", help="Whether to profile model execution.")] = False,
):
    """Run pure TensorRT inference for LightGlue model using Polygraphy (requires TensorRT to be installed)."""
    import numpy as np
    from polygraphy.backend.common import BytesFromPath
    from polygraphy.backend.trt import (
        CreateConfig,
        EngineFromBytes,
        EngineFromNetwork,
        NetworkFromOnnxPath,
        SaveEngine,
        TrtRunner,
    )

    from lightglue_dynamo import viz
    from lightglue_dynamo.preprocessors import DISKPreprocessor, SuperPointPreprocessor

    raw_images = [left_image_path, right_image_path]
    raw_images = [cv2.resize(cv2.imread(str(i)), (width, height)) for i in raw_images]
    images = np.stack(raw_images)
    match extractor_type:
        case Extractor.superpoint:
            images = SuperPointPreprocessor.preprocess(images)
        case Extractor.disk:
            images = DISKPreprocessor.preprocess(images)
    images = images.astype(np.float32)

    # Build TensorRT engine
    if model_path.suffix == ".engine":
        build_engine = EngineFromBytes(BytesFromPath(str(model_path)))
    else:  # .onnx
        build_engine = EngineFromNetwork(NetworkFromOnnxPath(str(model_path)), config=CreateConfig(fp16=fp16))
        build_engine = SaveEngine(build_engine, str(model_path.with_suffix(".engine")))

    with TrtRunner(build_engine) as runner:
        for _ in range(10 if profile else 1):  # Warm-up if profiling
            outputs = runner.infer(feed_dict={"images": images})
            keypoints, matches, mscores = outputs["keypoints"], outputs["matches"], outputs["mscores"]  # noqa: F841

        if profile:
            typer.echo(f"Inference Time: {runner.last_inference_time():.3f} s")

    viz.plot_images(raw_images)
    viz.plot_matches(keypoints[0][matches[..., 1]], keypoints[1][matches[..., 2]], color="lime", lw=0.2)
    if output_path is None:
        viz.plt.show()
    else:
        viz.save_plot(output_path)


if __name__ == "__main__":
    app()

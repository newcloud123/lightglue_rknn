import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "framework",
        type=str,
        choices=["torch", "ort"],
        help="The LightGlue framework to measure inference time. Options are 'torch' for PyTorch and 'ort' for ONNXRuntime.",
    )
    parser.add_argument(
        "--megadepth_path",
        type=Path,
        default=Path("megadepth_test_1500"),
        required=False,
        help="Path to the root of the MegaDepth dataset.",
    )
    parser.add_argument(
        "--img_size", type=int, default=1024, required=False, help="Image size."
    )
    parser.add_argument(
        "--extractor_type",
        type=str,
        choices=["superpoint", "disk"],
        default="superpoint",
        required=False,
        help="Type of feature extractor. Supported extractors are 'superpoint' and 'disk'.",
    )
    parser.add_argument(
        "--max_num_keypoints",
        type=int,
        default=512,
        required=False,
        help="Maximum number of keypoints to extract.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        required=False,
        help="cuda or cpu",
    )
    parser.add_argument(
        "--mp",
        action="store_true",
        help="Whether to enable mixed precision (CUDA only).",
    )
    parser.add_argument(
        "--flash",
        action="store_true",
        help="Whether to use Flash Attention (CUDA only).",
    )
    parser.add_argument(
        "--trt",
        action="store_true",
        help="Whether to use TensorRT (experimental).",
    )

    # ONNXRuntime-specific args
    parser.add_argument(
        "--extractor_path",
        type=str,
        default=None,
        required=False,
        help="Path to ONNX extractor model.",
    )
    parser.add_argument(
        "--lightglue_path",
        type=str,
        default=None,
        required=False,
        help="Path to ONNX LightGlue model.",
    )
    return parser.parse_args()


def get_megadepth_images(path: Path):
    sort_key = lambda p: int(p.stem.split("_")[0])
    images = sorted(
        list((path / "Undistorted_SfM/0015/images").glob("*.jpg")), key=sort_key
    ) + sorted(list((path / "Undistorted_SfM/0022/images").glob("*.jpg")), key=sort_key)
    return images


def create_models(
    framework: str,
    extractor_type="superpoint",
    max_num_keypoints=512,
    device="cuda",
    mp=False,
    flash=False,
    trt=False,
    extractor_path=None,
    lightglue_path=None,
):
    if framework == "torch":
        if extractor_type == "superpoint":
            extractor = (
                SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
            )
        elif extractor_type == "disk":
            extractor = DISK(max_num_keypoints=max_num_keypoints).eval().to(device)

        lightglue = LightGlue(extractor_type, mp=mp, flash=flash).eval().to(device)
    elif framework == "ort":
        sess_opts = ort.SessionOptions()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        if extractor_path is None:
            extractor_path = f"weights/{extractor_type}_{max_num_keypoints}.onnx"

        extractor = ort.InferenceSession(
            extractor_path,
            providers=providers,
        )

        if lightglue_path is None:
            lightglue_path = f"weights/{extractor_type}_lightglue.onnx"

        if trt:
            assert device == "cuda", "TensorRT is only supported on CUDA devices."
            providers = [
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_fp16_enable": True,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": "weights/cache",
                    },
                )
            ] + providers

        lightglue = ort.InferenceSession(
            lightglue_path,
            sess_options=sess_opts,
            providers=providers,
        )

    return extractor, lightglue


def measure_inference(
    framework: str, extractor, lightglue, image0, image1, device="cuda"
) -> float:
    if framework == "torch":
        # Feature extraction time is not measured
        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)

        # Measure only matching time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            result = lightglue({"image0": feats0, "image1": feats1})
        end.record()
        torch.cuda.synchronize()

        return start.elapsed_time(end)
    elif framework == "ort":
        # Feature extraction time is not measured
        kpts0, scores0, desc0 = extractor.run(None, {"image": image0})
        kpts1, scores1, desc1 = extractor.run(None, {"image": image1})

        lightglue_inputs = {
            "kpts0": LightGlueRunner.normalize_keypoints(
                kpts0, image0.shape[2], image0.shape[3]
            ),
            "kpts1": LightGlueRunner.normalize_keypoints(
                kpts1, image1.shape[2], image1.shape[3]
            ),
            "desc0": desc0,
            "desc1": desc1,
        }
        lightglue_outputs = ["matches0", "mscores0"]

        if device == "cuda":
            # Prepare IO-Bindings
            binding = lightglue.io_binding()

            for name, arr in lightglue_inputs.items():
                binding.bind_cpu_input(name, arr)

            for name in lightglue_outputs:
                binding.bind_output(name, "cuda")

            # Measure only matching time
            start = time.perf_counter()
            result = lightglue.run_with_iobinding(binding)
            end = time.perf_counter()
        else:
            start = time.perf_counter()
            result = lightglue.run(None, lightglue_inputs)
            end = time.perf_counter()

        return (end - start) * 1000


def evaluate(
    framework,
    megadepth_path=Path("megadepth_test_1500"),
    img_size=1024,
    extractor_type="superpoint",
    max_num_keypoints=512,
    device="cuda",
    mp=False,
    flash=False,
    trt=False,
    extractor_path=None,
    lightglue_path=None,
):
    images = get_megadepth_images(megadepth_path)
    image_pairs = list(zip(images[::2], images[1::2]))

    extractor, lightglue = create_models(
        framework=framework,
        extractor_type=extractor_type,
        max_num_keypoints=max_num_keypoints,
        device=device,
        mp=mp,
        flash=flash,
        trt=trt,
        extractor_path=extractor_path,
        lightglue_path=lightglue_path,
    )

    # Warmup
    for image0, image1 in image_pairs[:10]:
        image0, _ = load_image(str(image0), resize=img_size)
        image1, _ = load_image(str(image1), resize=img_size)

        if framework == "torch":
            image0 = image0[None].to(device)
            image1 = image1[None].to(device)
        elif framework == "ort" and extractor_type == "superpoint":
            image0 = rgb_to_grayscale(image0)
            image1 = rgb_to_grayscale(image1)

        _ = measure_inference(framework, extractor, lightglue, image0, image1, device)

    # Measure
    timings = []
    for image0, image1 in tqdm(image_pairs[10:]):
        image0, _ = load_image(str(image0), resize=img_size)
        image1, _ = load_image(str(image1), resize=img_size)

        if framework == "torch":
            image0 = image0[None].to(device)
            image1 = image1[None].to(device)
        elif framework == "ort" and extractor_type == "superpoint":
            image0 = rgb_to_grayscale(image0)
            image1 = rgb_to_grayscale(image1)

        inference_time = measure_inference(
            framework, extractor, lightglue, image0, image1, device
        )
        timings.append(inference_time)

    # Results
    timings = np.array(timings)
    print(f"Mean inference time: {timings.mean():.2f} +/- {timings.std():.2f} ms")
    print(f"Median inference time: {np.median(timings):.2f} ms")


if __name__ == "__main__":
    args = parse_args()
    if args.framework == "torch":
        import torch

        from lightglue import DISK, LightGlue, SuperPoint
        from lightglue.utils import load_image
    elif args.framework == "ort":
        import onnxruntime as ort

        from onnx_runner import LightGlueRunner, load_image, rgb_to_grayscale

    evaluate(**vars(args))

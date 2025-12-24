import sys
import numpy as np 
from rknn.api import RKNN
import cv2
import viz
DATASET_PATH = 'dataset.txt'
DEFAULT_RKNN_PATH = 'lightclue.rknn'
DEFAULT_QUANT = False
def preprocess(image: np.ndarray) -> np.ndarray:
    """Preprocess image from cv2.imread, (..., H, W, 3), BGR."""
    image = image[..., ::-1] / 255 * [0.299, 0.587, 0.114]
    image = image.sum(axis=-1, keepdims=True)  # (..., H, W, 1)
    axes = [*list(range(image.ndim - 3)), -1, -3, -2]
    image = image.transpose(*axes)  # (..., 1, H, W)
    return image
def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path [platform] [dtype(optional)] [output_rknn_path(optional)]".format(sys.argv[0]))
        print("       platform choose from [rk3562, rk3566, rk3568, rk3576, rk3588, rv1103, rv1106, rv1126b, rv1109, rv1126, rk1808]")
        print("       dtype choose from [i8, fp] for [rk3562, rk3566, rk3568, rk3576, rk3588, rv1103, rv1106, rv1126b]")
        print("       dtype choose from [u8, fp] for [rv1109, rv1126, rk1808]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ['i8', 'u8', 'fp']:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)
        elif model_type in ['i8', 'u8']:
            do_quant = True
        else:
            do_quant = False

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = DEFAULT_RKNN_PATH

    return model_path, platform, do_quant, output_path

if __name__ == '__main__':
    model_path, platform, do_quant, output_path = parse_arg()
    left_image_path = "IMG_0702.jpg"
    right_image_path = "IMG_0702.jpg"#"cut_jtport.tif"
    raw_images = [left_image_path, right_image_path]
    raw_images = [cv2.resize(cv2.imread(str(i)), (1024, 1024)) for i in raw_images]
    images = np.stack(raw_images)
    images = preprocess(images)
    images = images.astype(np.float16) # [2,1,1024,1024]
    print(images[0,0,0,0:100])
    print("input shape = ",images.shape)
    #import sys
    #sys.exit()
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    print('--> Config model')
    rknn.config(target_platform=platform,float_dtype='float16')
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')
      # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    #ret = rknn.init_runtime(target='rk3588')
    if ret != 0:
       print('Init runtime environment failed!')
       exit(ret)
    print('done')
    # Inference
    print('--> Running model')
    # outputs = rknn.inference(inputs=[img],data_format='nchw')
    #images = images.transpose(0,2,3,1)
    print("input images shape = ",images.shape)
    keypoints,matches,mscores = rknn.inference(inputs=[images],data_format='nchw')
    print("keypoints shape = ",keypoints.shape)
    print("matches shape = ",matches.shape)
    print("mscores shape = ",mscores.shape)
    print("mscores =  ",mscores)
    if False:
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
    print("keypoint[0] = ",keypoints[0]) 
    viz.plot_images(raw_images)
    viz.plot_matches(keypoints[0][matches[..., 1]], keypoints[1][matches[..., 2]], color="lime", lw=0.2)
    viz.save_plot("out.jpg")    


    # Release
    rknn.release()

import torch 
checkpoints_path = "/data/luoshiyong/code/LightGlue-ONNX-main/gim_lightglue_100h.ckpt"
superpint = "/data/luoshiyong/code/LightGlue-ONNX-main/superpoint_v1.pth"
lightglue = "/data/luoshiyong/code/LightGlue-ONNX-main/superpoint_lightglue.pth"
"""detector = SuperPoint({
            'max_num_keypoints': 2048,
            'force_num_keypoints': True,
            'detection_threshold': 0.0,
            'nms_radius': 3,
            'trainable': False,
        })
model = LightGlue({
    'filter_threshold': 0.1,
    'flash': False,
    'checkpointed': True,
})"""

state_dict = torch.load(checkpoints_path, map_location='cpu')
if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
for k in list(state_dict.keys()):
    if k.startswith('model.'):
        state_dict.pop(k)
    if k.startswith('superpoint.'):
        state_dict[k.replace('superpoint.', '', 1)] = state_dict.pop(k)
print(state_dict.keys())
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
state2 = torch.load(superpint, map_location='cpu')
print(state2.keys())


# detector.load_state_dict(state_dict)

state_dict = torch.load(checkpoints_path, map_location='cpu')
if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
for k in list(state_dict.keys()):
    if k.startswith('superpoint.'):
        state_dict.pop(k)
    if k.startswith('model.'):
        state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
print("state_dict")

# model.load_state_dict(state_dict)




"""
python dynamo.py export superpoint   --num-keypoints 1024   -b 2 -h 1024 -w 1024   -o weights/superpoint_lightglue_pipeline.onnx
python dynamo.py infer   weights/superpoint_lightglue_pipeline.onnx   assets/sacre_coeur1.jpg assets/sacre_coeur2.jpg   superpoint   -h 1024 -w 1024   --output out.jpg
"""
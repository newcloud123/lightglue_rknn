import onnx
from onnx import helper, TensorProto
from onnx import TensorShapeProto

# ===================== 1. 加载原始ONNX模型 =====================
onnx_path = "512_50.onnx"  # 替换为你的模型路径
onnx_model = onnx.load(onnx_path)
graph = onnx_model.graph

# 确认原始输入维度（NCHW [2,1,1024,1024]）
input_tensor = graph.input[0]
original_input_shape = []
for dim in input_tensor.type.tensor_type.shape.dim:
    original_input_shape.append(dim.dim_value)
print(f"原始输入维度（NCHW）：{original_input_shape}")
assert original_input_shape == [2,1,1024,1024], "原始输入维度必须是 [2,1,1024,1024]"

# ===================== 2. 修改模型输入为NHWC [2,1024,1024,1]（核心修复） =====================
# 修复：用del删除所有原有维度（替代clear()，兼容RepeatedCompositeContainer）
del input_tensor.type.tensor_type.shape.dim[:]

# 构造NHWC维度（batch=2, height=1024, width=1024, channel=1）
nhwc_dims = [2, 1024, 1024, 1]
for dim_val in nhwc_dims:
    dim = TensorShapeProto.Dimension()
    dim.dim_value = dim_val
    input_tensor.type.tensor_type.shape.dim.append(dim)

# ===================== 3. 插入Transpose节点（NHWC→NCHW）作为模型第一个操作 =====================
input_name = input_tensor.name
trans_output_name = f"{input_name}_nchw"

# 创建Transpose节点（perm=[0,3,1,2] 实现NHWC→NCHW）
trans_node = helper.make_node(
    op_type="Transpose",
    inputs=[input_name],
    outputs=[trans_output_name],
    name="NHWC2NCHW_Transpose",
    perm=[0, 3, 1, 2]
)

# 插入到模型最开头
graph.node.insert(0, trans_node)

# ===================== 4. 重定向后续节点的输入到Transpose输出 =====================
for node in graph.node[1:]:  # 跳过Transpose节点
    for i, inp in enumerate(node.input):
        if inp == input_name:
            node.input[i] = trans_output_name

# ===================== 5. 保存并验证模型 =====================
new_onnx_path = "model_nhwc_input_with_trans.onnx"
onnx.save(onnx_model, new_onnx_path)

# 验证修改结果
new_onnx = onnx.load(new_onnx_path)
new_input_shape = []
for dim in new_onnx.graph.input[0].type.tensor_type.shape.dim:
    new_input_shape.append(dim.dim_value)
print(f"对外暴露的输入维度（NHWC）：{new_input_shape}")  # 必输出 [2,1024,1024,1]

# 检查第一个节点是否为Transpose
first_node = new_onnx.graph.node[0]
print(f"模型第一个节点类型：{first_node.op_type}")  # 必输出 Transpose
print(f"Transpose节点perm参数：{first_node.attribute[0].ints}")  # 必输出 [0,3,1,2]

# 校验模型合法性
try:
    onnx.checker.check_model(new_onnx)
    print("✅ 模型修改成功！无任何错误，可直接转换RKNN")
except Exception as e:
    print(f"❌ 模型校验失败：{str(e)[:200]}")  # 只打印前200字符，避免过长

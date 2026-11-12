import onnx
from collections import Counter

MODEL_PATH = "/lab/models/resnet50.onnx"

# 1. Load the model file into memory
model = onnx.load(MODEL_PATH)

# 2. Validate it against the ONNX specification (raises an error if invalid)
onnx.checker.check_model(model)
print("Model is valid ONNX")

# 3. Basic information
print(f"Opset version : {model.opset_import[0].version}")
print(f"Total nodes   : {len(model.graph.node)}")

# 4. Inputs and outputs with their shapes
def shape_of(tensor):
	dims = []
	for d in tensor.type.tensor_type.shape.dim:
		# dim_param = named (dynamic) dimension, dim_value = fixed size
		dims.append(d.dim_param if d.dim_param else d.dim_value)
	return dims

print("\nInputs:")
for t in model.graph.input:
	print(f"   {t.name}: {shape_of(t)}")

print("\nOutputs:")
for t in model.graph.output:
        print(f"   {t.name}: {shape_of(t)}")

# 5. Which operation types the graph contains, and how many of each
print("\nOperation types:")
counts = Counter(node.op_type for node in model.graph.node)
for op, n in counts.most_common():
	print(f"  {op:<20}  {n}")

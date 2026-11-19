import sys
import numpy as np
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner

# 1. Engine files from the command line e.g. r50_fp16.plan
engine_path = "/lab/models/" + sys.argv[1]
runner = TrtRunner(EngineFromBytes(BytesFromPath(engine_path)))

# 2. Preprocessed validation data (from 09_prep_val.py)
images = np.load("/lab/models/val_images.npy")
labels = np.load("/lab/models/val_labels.npy")

# 3. Predict in batches of 32 and count correct answers
correct = 0
with runner:
	dtype = runner.get_input_metadata()["input"].dtype
	print("Engine input dtype:", dtype)
	for i in range(0, len(images), 32):
		logits = runner.infer({"input": images[i:i+32].astype(dtype)})["logits"]
		correct += (logits.argmax(axis=1) == labels[i:i+32]).sum()
print(f"Correct: {correct} / {len(images)}")
print(f"Accuracy: {100* correct / len(images)}%")

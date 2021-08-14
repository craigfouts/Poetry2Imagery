from transformers import convert_graph_to_onnx
from pathlib import Path

convert_graph_to_onnx.convert(
        framework='pt',
        model="pytorch_output",
        output=Path("model/model.onnx"),
        opset=12,
        tokenizer="pytorch_output",
        pipeline_name="sentiment-analysis")


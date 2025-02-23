import pyvista as pv
import torch 
import torch.nn as nn 
from collections import OrderedDict 
import json
import os 
import sys 
import importlib.util 
import argparse

class ModelSummary:
    def __init__(self, model_path: str, model_class_name: str = 'Model', input_shape: tuple = (3, 224, 224), device: str = 'cuda'):
        """
        初始化 ModelSummary 类，加载模型并准备获取模型的层结构。

        Args:
            model_path (str): 模型文件的路径。
            model_class_name (str): 模型类的名称，默认为 'Model'。
            input_shape (tuple): 输入形状，默认为 (3, 224, 224)。
            device (str): 使用的设备，'cuda', 'cpu', 或 'mps'，默认为 'cuda'。
        """
        self.model_path = model_path
        self.model_class_name = model_class_name
        self.input_shape = input_shape
        self.device = device.lower()
        assert self.device in ["cuda", "cpu", "mps"], "Input device is not valid, please specify 'cuda', 'cpu', or 'mps'"
        self.model = self.load_model()

    def summary(self, model, input_size, batch_size=-1):
        """
        获取模型的概述信息（包括每层的输入输出形状和参数数量）。
        """
        dtype = (
            torch.cuda.FloatTensor if self.device == "cuda" and torch.cuda.is_available() else torch.FloatTensor
        )

        if isinstance(input_size, tuple):
            input_size = [input_size]

        x = [
            torch.randn(2, *in_size).type(dtype).to(self.device)
            for in_size in input_size
        ]

        summary_dict = OrderedDict()
        hooks = []

        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]

                if class_name == 'LeNet':
                    return

                m_key = f"{class_name}-{len(summary_dict) + 1}"

                summary_dict[m_key] = OrderedDict()
                summary_dict[m_key]["type"] = class_name

                if isinstance(input, tuple):
                    input_shape = [list(inp.size()) for inp in input]
                else:
                    input_shape = list(input[0].size())

                if batch_size != -1:
                    if isinstance(input_shape, list):
                        for shape in input_shape:
                            shape[0] = batch_size
                    else:
                        input_shape[0] = batch_size

                summary_dict[m_key]["input_shape"] = input_shape

                if isinstance(output, (list, tuple)):
                    output_shape = [list(o.size()) for o in output]
                else:
                    output_shape = list(output.size())

                if batch_size != -1:
                    if isinstance(output_shape, list):
                        for shape in output_shape:
                            shape[0] = batch_size
                    else:
                        output_shape[0] = batch_size

                summary_dict[m_key]["output_shape"] = output_shape
                summary_dict[m_key]["params"] = sum(p.numel() for p in module.parameters())

            if not isinstance(module, (nn.Sequential, nn.ModuleList)):
                hooks.append(module.register_forward_hook(hook))

        model.apply(register_hook)
        model(*x)
        for h in hooks:
            h.remove()

        return summary_dict

    def load_model(self) -> torch.nn.Module:
        """
        动态加载模型文件并实例化模型。
        """
        model_dir = os.path.dirname(self.model_path)
        sys.path.append(model_dir)

        model_name = os.path.splitext(os.path.basename(self.model_path))[0]
        spec = importlib.util.spec_from_file_location(model_name, self.model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, self.model_class_name):
            raise ImportError(f"Model class '{self.model_class_name}' not found in {self.model_path}")

        model_class = getattr(module, self.model_class_name)
        model = model_class()
        return model

    def get_summary_json(self) -> str:
        """
        获取并返回模型的层结构信息的 JSON 格式。
        """
        device = torch.device(self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu")
        self.model.to(device)

        if isinstance(self.input_shape, tuple):
            input_sizes = [self.input_shape]
        else:
            input_sizes = self.input_shape

        summary_dict = self.summary(self.model, input_sizes)
        return json.dumps(summary_dict, indent=4)

class LayerVisualizer:
    def __init__(self, layers: dict, spacing: float = 10):
        """
        初始化 LayerVisualizer 类，设置层信息和可视化参数。

        Args:
            layers (dict): 模型的层信息，通常是通过 summary 函数获取的字典。
            spacing (float): 层与层之间的间距，默认为 10。
        """
        self.layers = layers
        self.spacing = spacing

    def calculate_layer_position(self):
        """
        计算每一层的位置，基于每层的输入输出形状。
        """
        layer_names = list(self.layers.keys())
        n = len(layer_names)
        
        center_idx = n // 2
        center_layer_name = layer_names[center_idx]
        center_layer_input_shape = self.layers[center_layer_name]["input_shape"]
        center_layer_output_shape = self.layers[center_layer_name]["output_shape"]

        layer_positions = {}
        layer_positions[center_layer_name] = {
            "input_shape": center_layer_input_shape,
            "output_shape": center_layer_output_shape,
            "position": (0, 0, 0)
        }

        current_pos = -self.spacing
        for i in range(center_idx - 1, -1, -1):
            layer_positions[layer_names[i]] = {
                "input_shape": self.layers[layer_names[i]]["input_shape"],
                "output_shape": self.layers[layer_names[i]]["output_shape"],
                "position": (current_pos, 0, 0)
            }
            current_pos -= self.spacing

        current_pos = self.spacing
        for i in range(center_idx + 1, n):
            layer_positions[layer_names[i]] = {
                "input_shape": self.layers[layer_names[i]]["input_shape"],
                "output_shape": self.layers[layer_names[i]]["output_shape"],
                "position": (current_pos, 0, 0)
            }
            current_pos += self.spacing

        return layer_positions

    def generate_layers(self, position, layer_shape):
        """
        生成层的可视化图形，返回 `pyvista` 图形对象。
        """
        if len(layer_shape) == 4:
            channels, height, width = layer_shape[1], layer_shape[2], layer_shape[3]
        elif len(layer_shape) == 3:
            channels, height, width = 1, layer_shape[1], layer_shape[2]
        elif len(layer_shape) == 2:
            if layer_shape[1] >= 50:
                channels, height, width = 50, 1, 1
            else:
                channels, height, width = layer_shape[1], 1, 1

        layers = []
        for i in range(channels):
            center = (position[0] + (i - channels // 2) * 0.1, position[1], position[2])
            layer = pv.Plane(center=center, direction=(1, 0, 0), i_size=height, j_size=width, i_resolution=1, j_resolution=1)
            layers.append(layer)
        layers = pv.MultiBlock(layers)

        label_height = width // 2 + 2
        return layers, label_height

    def generate_label(self, layer_name, input_shape, output_shape, position, label_height):
        """
        生成层的标签。
        """
        label = f"{layer_name}\n{input_shape} -> {output_shape}"
        label_position = (position[0], position[1] - 1, label_height)
        label = pv.Label(label, position=label_position, size=12)
        return label

    def visualize_layers(self):
        """
        可视化模型的每一层，展示输入输出形状，并连接层与层之间的结构。
        """
        plotter = pv.Plotter()
        plotter.show_axes()

        layer_positions = self.calculate_layer_position()

        for layer_name, layer_info in layer_positions.items():
            layer_name = layer_name.split("-")[0]
            input_shape = layer_info["input_shape"][0]
            output_shape = layer_info["output_shape"]
            position = layer_info["position"]

            # 生成层的图形
            layer, label_height = self.generate_layers(position, output_shape)
            plotter.add_mesh(layer, color='w', show_edges=True)
            # 生成层的标签
            label = self.generate_label(layer_name, input_shape, output_shape, position, label_height)
            plotter.add_actor(label)

        plotter.show()

def main():
    parser = argparse.ArgumentParser(description="Load a PyTorch model and visualize its layers.")
    parser.add_argument('model_path', type=str, help="Path to the model file.")
    parser.add_argument('--model_class_name', type=str, default='Model', help="Name of the model class.")
    parser.add_argument('--input_shape', type=int, nargs=3, default=[3, 224, 224], help="Input shape of the model (C, H, W).")
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'], default='cuda', help="Device to run the model on.")

    args = parser.parse_args()

    model_summary = ModelSummary(model_path=args.model_path, model_class_name=args.model_class_name, input_shape=tuple(args.input_shape), device=args.device)
    summary_json = model_summary.get_summary_json()
    layers = json.loads(summary_json)

    visualizer = LayerVisualizer(layers)
    visualizer.visualize_layers()

if __name__ == "__main__":
    main()

# 示例
# python visual.py model.py --model_class_name LeNet --input_shape 1 28 28 --device cpu
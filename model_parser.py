import torch
import torchvision.models as models
import torch.nn as nn

def create_model(model_name):
    model_dict = {
        # AlexNet
        'alexnet': models.alexnet,
        'alexnet_weights': models.alexnet,

        # ConvNeXt
        'convnext_base': models.convnext_base,
        'convnext_large': models.convnext_large,
        'convnext_small': models.convnext_small,
        'convnext_tiny': models.convnext_tiny,

        # DenseNet
        'densenet121': models.densenet121,
        'densenet161': models.densenet161,
        'densenet169': models.densenet169,
        'densenet201': models.densenet201,

        # EfficientNet
        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b1': models.efficientnet_b1,
        'efficientnet_b2': models.efficientnet_b2,
        'efficientnet_b3': models.efficientnet_b3,
        'efficientnet_b4': models.efficientnet_b4,
        'efficientnet_b5': models.efficientnet_b5,
        'efficientnet_b6': models.efficientnet_b6,
        'efficientnet_b7': models.efficientnet_b7,
        'efficientnet_v2_l': models.efficientnet_v2_l,
        'efficientnet_v2_m': models.efficientnet_v2_m,
        'efficientnet_v2_s': models.efficientnet_v2_s,

        # GoogLeNet
        'googlenet': models.googlenet,
        'googlenet_weights': models.googlenet,

        # InceptionV3
        'inception_v3': models.inception_v3,
        'inception_v3_weights': models.inception_v3,

        # MNASNet
        'mnasnet0_5': models.mnasnet0_5,
        'mnasnet0_75': models.mnasnet0_75,
        'mnasnet1_0': models.mnasnet1_0,
        'mnasnet1_3': models.mnasnet1_3,

        # MaxVit
        'maxvit_t': models.maxvit_t,

        # MobileNetV2
        'mobilenet_v2': models.mobilenet_v2,
        'mobilenet_v3_large': models.mobilenet_v3_large,
        'mobilenet_v3_small': models.mobilenet_v3_small,

        # RegNet
        'regnet_x_16gf': models.regnet_x_16gf,
        'regnet_x_1_6gf': models.regnet_x_1_6gf,
        'regnet_x_32gf': models.regnet_x_32gf,
        'regnet_x_3_2gf': models.regnet_x_3_2gf,
        'regnet_x_400mf': models.regnet_x_400mf,
        'regnet_x_800mf': models.regnet_x_800mf,
        'regnet_x_8gf': models.regnet_x_8gf,
        'regnet_y_128gf': models.regnet_y_128gf,
        'regnet_y_16gf': models.regnet_y_16gf,
        'regnet_y_1_6gf': models.regnet_y_1_6gf,
        'regnet_y_32gf': models.regnet_y_32gf,
        'regnet_y_3_2gf': models.regnet_y_3_2gf,
        'regnet_y_400mf': models.regnet_y_400mf,
        'regnet_y_800mf': models.regnet_y_800mf,
        'regnet_y_8gf': models.regnet_y_8gf,

        # ResNet
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,

        # ResNeXt
        'resnext50_32x4d': models.resnext50_32x4d,
        'resnext101_32x8d': models.resnext101_32x8d,

        # ShuffleNetV2
        'shufflenet_v2_x0_5': models.shufflenet_v2_x0_5,
        'shufflenet_v2_x1_0': models.shufflenet_v2_x1_0,
        'shufflenet_v2_x1_5': models.shufflenet_v2_x1_5,
        'shufflenet_v2_x2_0': models.shufflenet_v2_x2_0,

        # SqueezeNet
        'squeezenet1_0': models.squeezenet1_0,
        'squeezenet1_1': models.squeezenet1_1,

        # Swin Transformer
        'swin_b': models.swin_b,
        'swin_s': models.swin_s,
        'swin_t': models.swin_t,
        'swin_v2_b': models.swin_v2_b,
        'swin_v2_s': models.swin_v2_s,
        'swin_v2_t': models.swin_v2_t,

        # VGG
        'vgg11': models.vgg11,
        'vgg11_bn': models.vgg11_bn,
        'vgg13': models.vgg13,
        'vgg13_bn': models.vgg13_bn,
        'vgg16': models.vgg16,
        'vgg16_bn': models.vgg16_bn,
        'vgg19': models.vgg19,
        'vgg19_bn': models.vgg19_bn,

        # Vision Transformer
        'vit_b_16': models.vit_b_16,
        'vit_b_32': models.vit_b_32,
        'vit_h_14': models.vit_h_14,
        'vit_l_16': models.vit_l_16,
        'vit_l_32': models.vit_l_32,

        # Wide ResNet
        'wide_resnet50_2': models.wide_resnet50_2,
        'wide_resnet101_2': models.wide_resnet101_2
    }

    if model_name not in model_dict:
        return {'error': 'Model not found'}
    
    return model_dict[model_name](weights=None)

def get_model_structure(model_name, input_shape=(1, 3, 224, 224)):
    # 创建模型
    try:
        model = create_model(model_name)
    except ValueError as e:
        return {"error": str(e)}

    model.eval()

    layer_shapes = [{'name': 'input', 'input_shape': input_shape, 'output_shape': input_shape, 'params': 0}]
    hooks = []
    
    def get_layer_params(module):
        return sum(p.numel() for p in module.parameters())

    def hook_fn(module, input, output):
        layer_shapes.append({
            'name': module.__class__.__name__,
            'input_shape': tuple(input[0].shape),
            'output_shape': tuple(output.shape),
            'params': get_layer_params(module)
        })

    def register_hooks(module):
        if isinstance(module, nn.Sequential):
            for layer in module.children():
                register_hooks(layer)
        else:
            hooks.append(module.register_forward_hook(hook_fn))

    for layer in model.children():
        register_hooks(layer)

    dummy_input = torch.randn(*input_shape)
    model(dummy_input)

    for hook in hooks:
        hook.remove()

    return layer_shapes

if __name__ == '__main__':
    model_name = 'convnext_base'
    input_shape = (1, 3, 224, 224)
    layer_shapes = get_model_structure(model_name, input_shape)
    
    for layer in layer_shapes:
        print(layer)
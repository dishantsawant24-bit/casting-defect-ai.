# Local model checkpoint

Place the approved trained checkpoint at `models/resnet18_best.pth`. Model weights are excluded from Git by default because the binary is approximately 44.8 MB and redistribution permission must be confirmed. The loader expects a 12-class ResNet18 checkpoint with a `model_state_dict` entry and uses PyTorch `weights_only=True`.

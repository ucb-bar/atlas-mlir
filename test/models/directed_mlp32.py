"""Public 32x32 Linear/ReLU/Linear capture with exact finite values.

The loader is independent of the Atlas compiler. Its weights and biases are
ordinary PyTorch parameters; the compiler must discover their graph roles and
pack them through the capture manifest.
"""

import torch


def get_model_and_inputs():
    first = torch.nn.Linear(32, 32)
    second = torch.nn.Linear(32, 32)
    with torch.no_grad():
        first.weight.zero_()
        second.weight.zero_()
        for column in range(32):
            first.weight[column, (column + 3) % 32] = 1.0
            second.weight[column, (column + 5) % 32] = 1.0
        first.bias.copy_(torch.tensor(
            [0.5 if column % 2 == 0 else 1.0 for column in range(32)]
        ))
        second.bias.copy_(torch.tensor(
            [0.25 if column % 2 == 0 else 0.5 for column in range(32)]
        ))
    model = torch.nn.Sequential(first, torch.nn.ReLU(), second).eval()
    values = (0.0, 0.5, 1.0, 2.0, -1.0)
    source = torch.tensor(
        [values[(row * 7 + col * 3) % len(values)]
         for row in range(32) for col in range(32)],
        dtype=torch.float32,
    ).reshape(32, 32)
    return model, (source,)

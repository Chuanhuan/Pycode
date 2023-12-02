import torch
import torch.nn as nn


# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        breakpoint()  # Flatten the output for the fully connected layer
        x = self.relu(x)
        x = self.fc(x)
        return x


# Instantiate the model
model = SimpleCNN()


# Define a hook function to print gradients during backward pass
def backward_hook_fn(module, grad_input, grad_output):
    print(f"{module.__class__.__name__} grad_input: {grad_input}")
    print(f"{module.__class__.__name__} grad_output: {grad_output}")


# Define a hook function to print activations during forward pass
def forward_hook_fn(module, input, output):
    print(f"{module.__class__.__name__} input: {input}")
    print(f"{module.__class__.__name__} output: {output}")


# breakpoint()

# Attach hooks to the model
backward_hook = model.fc.register_backward_hook(backward_hook_fn)
forward_hook = model.fc.register_forward_hook(forward_hook_fn)

# Dummy input tensor
dummy_input = torch.randn(1, 3, 32, 32)
dummy_target = torch.randint(0, 10, (1,))

# Forward and backward passes with hooks
output = model(dummy_input)
loss = nn.CrossEntropyLoss()(output, dummy_target)
loss.backward()

# Detach the hooks after using them
backward_hook.remove()
forward_hook.remove()

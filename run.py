import torch
import torch.nn as nn
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)

class ComplexEncodingTest(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ComplexEncodingTest, self).__init__()
        # PyTorch natively allocates complex weights using torch.cfloat
        self.complex_linear = nn.Parameter(
            torch.randn(output_dim, input_dim, dtype=torch.cfloat)
        )
        
    def complex_relu(self, z):
        # Cardioid/cReLU variant: Apply activation to magnitude or parts
        # Here we apply standard ReLU to the real and imaginary parts independently
        return torch.complex(torch.relu(z.real), torch.relu(z.imag))

    def forward(self, z):
        # Matrix multiplication in the complex domain
        # (A + iB)(C + iD) = (AC - BD) + i(AD + BC)
        out = torch.matmul(z, self.complex_linear.t())
        return self.complex_relu(out)

# 1. Create Synthetic Input: [Batch, Features]
# Let's say we have 2 samples, 3 features each.
# Magnitudes represent raw content features (e.g., frequencies or pixel scales)
magnitudes = torch.tensor([[1.0, 2.0, 3.0], 
                           [1.0, 2.0, 3.0]], dtype=torch.float32)

# Phases represent structural context (Sample 1 is 0° phase, Sample 2 is 90° phase)
phases = torch.tensor([[0.0, 0.0, 0.0], 
                       [np.pi/2, np.pi/2, np.pi/2]], dtype=torch.float32)

# 2. Encode into the complex plane via Euler's identity: z = r * e^(i*theta)
real_part = magnitudes * torch.cos(phases)
imag_part = magnitudes * torch.sin(phases)
complex_inputs = torch.complex(real_part, imag_part)

# 3. Pass through our Complex Layer
model = ComplexEncodingTest(input_dim=3, output_dim=2)
output = model(complex_inputs)

print("--- Complex Inputs ---")
print(complex_inputs)
print("\n--- Model Output (Complex Plane) ---")
print(output)
print("\n--- Output Phases (Angles in Radians) ---")
print(torch.angle(output))
from model import TFN
from torch.autograd import Variable
import torch

model = TFN((46, 74, 300), (32, 32, 128), 64, (0.3, 0.3, 0.3, 0.3), 32)

x_a = Variable( torch.randn(16, 46) )
x_v = Variable( torch.randn(16, 74) )
x_t = Variable( torch.randn(16, 20, 300) )

y = model(x_a, x_v, x_t)

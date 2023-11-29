from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes

t1 = Tensor.arange(start=3, stop=9, step=1, dtype=dtypes.float16)
t2 = Tensor.randn(6, 1, dtype=dtypes.float16) # randn - random distribution
t3 = Tensor.rand(1, 6, dtype=dtypes.float16) # rand - uniform distribution

t4 = t2.matmul(t3) # matrix multiplication

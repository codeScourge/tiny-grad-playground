from tinygrad.nn.state import safe_save, get_state_dict
from tinygrad.nn.state import get_parameters
from tinygrad.nn.optim import SGD
from tinygrad.helpers import dtypes
from tinygrad import Tensor

from extra.datasets import fetch_mnist  # located in tinygrad repo

import numpy as np


LR = 3e-4
EPOCHS = 1000
BATCH_SIZE = 64
MODEL_PATH = "model.safetensors"

def sparse_categorical_crossentropy(self, Y, ignore_index=-1) -> Tensor:
    loss_mask = Y != ignore_index
    y_counter = Tensor.arange(self.shape[-1], dtype=dtypes.int32, requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
    y = ((y_counter == Y.flatten().reshape(-1, 1)).where(-1.0, 0) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
    return self.log_softmax().mul(y).sum() / loss_mask.sum()

class Linear:
    def __init__ (self, input_size:int, output_size:int, bias:bool=True, initialization: str="kaiming_uniform"):

        # would be the same as Tensor.kaiming_uniform(input_size, output_size)
        self.weight = getattr(Tensor, initialization)(output_size, input_size)
        self.bias = Tensor.zeros(output_size) if bias else None

    def __call__(self, tensor):
        return tensor.linear(self.weight.transpose(), self.bias)


class Tinynet:
    """
    2 layer NN with leaky RelU activation function  to classify MNIST digits
    """

    def __init__(self):
        self.l1 = Linear(784, 128, bias=False)
        self.l2 = Linear(128, 10, bias=False)

    def __call__(self, x):
        x = self.l1(x)
        x = x.leakyrelu()
        x = self.l2(x)
        return x


net = Tinynet()
optimizer = SGD(get_parameters(net), lr=LR) # get_parameters returns the same as [net.l1, net.l2]


x_train, y_train, x_test, y_test = fetch_mnist()

with Tensor.train():
    for epoch in range(EPOCHS):
        idxs = np.random.randint(0, x_train.shape[0], BATCH_SIZE)
        batch = Tensor(x_train[idxs], requires_grad=False)
        labels = Tensor(y_train[idxs])

        # forward pass
        logits = net(batch)
        loss = sparse_categorical_crossentropy(logits, labels)

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # predictions and probability
        pred = logits.argmax(axis=-1)
        acc = (pred == labels).mean()

        if epoch % 100 == 0:
            print(f"Step {epoch+1} | Loss: {loss.numpy()} | Accuracy: {acc.numpy()}")


# saving the model
state_dict = get_state_dict(net)
safe_save(state_dict, MODEL_PATH)

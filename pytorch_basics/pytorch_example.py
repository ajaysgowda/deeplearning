"""
pytorch starter code.
an example on how to use pytorch.
not meant to run
"""
# access to layers
import torch.nn as nn

# access to activation functions eg. relu, sigmoid, tanh
import torch.nn.functional as Func

# access to optimizers like adam
import torch.optim as optim

# access to main module
import torch as T


class LinearClassifier(nn.Module):
    """
    Example LinearClassifier
    Derived class from base class nn.Module.
    This gives access to deep NN parameters
    """
    def __init__(self, lr, n_classes, input_dims):
        super(LinearClassifier, self).__init__()

        # declaring layers for the classifier
        # 3 Linear Layers

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_classes)

        # self.parameters() is from base class
        # below statement states that we are optimizing to self.parameters
        # with a learning rate of lr
        self.optimizers = optim.Adam(self.parameters(), lr=lr)

        # set loss function
        self.loss = nn.CrossEntropyLoss()

        # device to use for calculation
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # send entire network to the device
        # ensure the tensors are compatible with the selected device
        self.to(self.device)

    def forward(self, data):  # pylint: disable=arguments-differ
        """
        Forward propogation algorithm
        """
        layer1 = Func.sigmoid(self.fc1(data))
        layer2 = Func.sigmoid(self.fc2(layer1))
        layer3 = self.fc3(layer2)

        return layer3

    def learn(self, data, labels):
        """
        Learn function
        """
        # zero-out gradients for the optimizer
        # pytorch keeps track of gradients of learning loops
        self.optimizer.zero_grad()

        # convert dat to tensors
        data = T.tensor(data=data).to(self.device)
        labels = T.tensor(data=labels).to(self.device)

        predictions = self.forward(data)

        cost = self.loss(predictions, labels)

        # important!!
        cost.backward()

        self.optimizer.step()

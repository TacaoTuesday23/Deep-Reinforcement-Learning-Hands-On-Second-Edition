import torch
import torch.nn as nn

class OurModule(nn.Module):
    def __init__(self, num_inputs, num_classes, dropout_prob=0.3):
        # this is used to inherit all the properties and methods of nn.Module
        super(OurModule, self).__init__()
        # pipe is a new attribute of this subclass
        # it does not depenod on the input parameters, so it is not passed to the constructor
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs, 5),
            nn.ReLU(),
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
            nn.Dropout(p=dropout_prob),
            nn.Softmax(dim=1)
        )

    # foward is a special method, it is called when the object is called as a function
    # but for normal methods, the format is self.method_name(self, arg1, arg2, ...)
    def forward(self, x):
        return self.pipe(x)

if __name__ == "__main__":
    net = OurModule(num_inputs=2, num_classes=3)
    print(net)
    v = torch.FloatTensor([[2, 3]])
    # we just call the object as a function, it will call the forward method
    out = net(v)
    print(out)
    print("Cuda's availability is %s" % torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Data from cuda: %s" % out.to('cuda'))

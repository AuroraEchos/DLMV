from torch import nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1  = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2  = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1  = nn.Linear(in_features=64*4*4, out_features=512)
        self.fc2  = nn.Linear(in_features=512, out_features=10)
        self.relu  = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) 
        x = self.pool(self.relu(self.conv2(x))) 
        x = x.view(-1,  64*4*4)
        x = self.relu(self.fc1(x)) 
        x = self.fc2(x) 
        return x

if __name__ == '__main__':
    model = LeNet()
    print(model)

#Import library
from torch import nn


#...............................................
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,padding=1,stride=1)
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1,stride=1)
        self.conv3=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,stride=1)
        #self.conv4=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,stride=1)
        #self.conv5=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,stride=1)

        self.pool = nn.MaxPool2d(kernel_size=2,stride=1)

        self.fc1=nn.Linear(23*23*32, 512)
        self.fc2=nn.Linear(512,256)
        self.out=nn.Linear(256,10)
        self.dropout=nn.Dropout(0.2)
    
    def forward(self, x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=self.pool(F.relu(self.conv3(x)))
        #x=self.pool(F.relu(self.conv4(x)))
        #x=self.pool(F.relu(self.conv5(x)))

        x=x.view(-1,23*23*32)
        x=self.dropout(x)
        x=self.dropout(F.relu(self.fc1(x)))
        x=self.dropout(F.relu(self.fc2(x)))
        x=self.out(x)

        return x


class NN(nn.Module):
    def __init__(self, input, hidden, output, learning_rate):
        super(threelayerNet, self).__init__()

        self.input=input
        self.hidden=hidden
        self.output=output
        
        self.lr=learning_rate

        #fully connected layer
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, output)

        #activation function
        self.activation = nn.Sigmoid()

        self.dropout=nn.Dropout(0.2)


    def forward(self, x):
        x=x.view(-1, 28*28)
        x=self.activation(self.fc1(x))
        x=self.activation(self.fc2(x))

        return x


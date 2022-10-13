from torch import nn


class NN(nn.Module):
    def __init__(self, num_classes, fc_dims):
        super(NN, self).__init__()
        assert len(fc_dims) > 0, 'fc_dims can not be empty'

        fcs = []

        for i in range(len(fc_dims) - 1):
            fcs.append(
                nn.Sequential(
                    nn.Linear(fc_dims[i], fc_dims[i + 1]),
                    nn.BatchNorm1d(fc_dims[i + 1]),
                    nn.ReLU(inplace=True)
                )
            )
        fcs.append(nn.Linear(fc_dims[-1], num_classes))

        self.fc = nn.Sequential(*fcs)
        self.softmax = nn.Softmax(dim=-1)

        # self.input=input
        # self.hidden=hidden
        # self.output=output
        
        # self.lr=learning_rate

        # #fully connected layer
        # self.fc1 = nn.Linear(input, hidden)
        # self.fc2 = nn.Linear(hidden, output)

        # #activation function
        # self.activation = nn.Sigmoid()

        # self.dropout=nn.Dropout(0.2)


    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.softmax(x)

        # x=x.view(-1, 28*28)
        # x=self.activation(self.fc1(x))
        # x=self.activation(self.fc2(x))

        return x
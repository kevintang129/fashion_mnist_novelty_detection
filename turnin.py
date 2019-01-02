# 1: Preprocessing

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

# 2: Model

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.sigmoid(F.max_pool2d(self.conv1(x), 2))
        x = torch.sigmoid(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.sigmoid(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

# 3: Postprocess
    prob, outputs = torch.max(outputs,1)#[1]
    maxi = torch.max(prob)
    mini = torch.min(prob)
    prob = torch.add(prob,mini*-1)
    prob = torch.div(prob,maxi-mini)
    y = torch.ones(prob.shape)
    z = torch.zeros(prob.shape)
    outputs = torch.where(prob <= .4, y, z)
# 4: Written explanation
We implemented a CNN with two convolutions and then trained it on the training data, 
wherein we used a sigmoid activation function. The softmax function was used to obtain 
classification probabilities. An L1 loss function was implemented following this, 
along with stochastic gradient descent. During the post process, we examined the outputs, 
which were tensors of probabilities. If any probability was under our proposed threshold (see above), 
it was likely representative of a novel piece of data.
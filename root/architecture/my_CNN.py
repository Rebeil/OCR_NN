import torch.nn as nn


class ConvNet(nn.Module):
    """_summary_
    architecture NN
    """
    def __init__(self):
        super().__init__()
        #region свёрточные слои
        self.act = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv0 = nn.Conv2d(1, 128, kernel_size=3 , stride=1, padding=0)
        self.conv1 = nn.Conv2d(128, 128,kernel_size=3 , stride=1, padding=0)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3 , stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        #self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        #endregion
        
                
        self.adaptivepool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 33)
        
    def forward(self, tensor_image):
        """_summary_

        Args: tensor_image - tensor
            tensor_image (_type_): _description_

        Returns: tensor.size = 33
            _type_: _description_
        """
        #print(x.shape)
        out = self.conv0(tensor_image)
        out = self.act(out)
        out = self.maxpool(out)
        #print(out.shape)
        out = self.conv1(out)
        out = self.act(out)
        out = self.maxpool(out)
        #print(out.shape)
        out = self.conv2(out)
        out = self.act(out)
        out = self.maxpool(out)
        #print(out.shape)
        out = self.conv3(out)
        out = self.act(out)
        out = self.maxpool(out)
        #print(out.shape)
        
                
        out = self.adaptivepool(out)
        out = self.flatten(out)
        out = self.linear1(out)
        #print(out.shape)
        out = self.act(out)
        out = self.linear2(out)
        #print(out.shape)
        
        
        return out
        
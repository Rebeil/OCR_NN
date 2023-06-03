import torch.nn as nn


class ConvNet(nn.Module):       
    """
        Args:
            in_channels (int): кол-во каналов \n
            num_channels (int): уол-вл свёрток \n
            out_channels (int): кол-во выходных классов \n
    """
    def __init__(self,in_channels:int=1, num_channels:int=128, out_channels:int=33):

        super().__init__()
        #region свёрточные слои
        self.act = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv0 = nn.Conv2d(in_channels, num_channels, kernel_size=3 , stride=1, padding=0)
        self.conv1 = nn.Conv2d(num_channels, num_channels,kernel_size=3 , stride=1, padding=0)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3 , stride=1, padding=0)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=0)
        #self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        #endregion
        
                
        self.adaptivepool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(num_channels, num_channels//2)
        self.linear2 = nn.Linear(num_channels//2, out_channels)
        
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
        
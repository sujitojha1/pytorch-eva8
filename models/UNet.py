import torch.nn as nn

def ConvBlock(in_channels, out_channels):
    """
    Creates a convolution block with two Conv2d-BatchNorm2d-ReLU layers.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        
    Returns:
        nn.Sequential: The created convolution block.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class ContractingBlock(nn.Module):
    """
    A contracting block module for use in a U-Net architecture.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Convolution block with two Conv2d-BatchNorm2d-ReLU layers
        self.convblock = ConvBlock(in_channels, out_channels)

        # Max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Forward pass through the contracting block.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after max pooling.
            torch.Tensor: Output tensor before max pooling (skip connection).
        """

        x = self.convblock(x)

        # Save the output before max pooling as the skip connection
        skip = x
        x = self.maxpool(x)

        return x, skip


class ExpandingBlock(nn.Module):
    """
    An expanding block module for use in a U-Net architecture.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Convolution block with two Conv2d-BatchNorm2d-ReLU layers
        self.convblock = ConvBlock(in_channels, out_channels)

        # Upsampling layer using a transposed convolution
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)

    def forward(self, x, skip):
        """
        Forward pass through the expanding block.
        
        Args:
            x (torch.Tensor): Input tensor.
            skip (torch.Tensor): Skip connection tensor.
        
        Returns:
            torch.Tensor: Output tensor after concatenating the skip connection.
        """
        x = self.convblock(x)

        x = self.upsample(x)

        # Concatenate the skip connection
        x = torch.cat((x, skip), dim=1)

        return x

class UNet(nn.Module):
    """
    A U-Net architecture for image segmentation.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Contracting path
        self.contract1 = ContractingBlock(in_channels, 64)
        self.contract2 = ContractingBlock(64, 128)
        self.contract3 = ContractingBlock(128, 256)
        self.contract4 = ContractingBlock(256, 512)

        # Expanding path
        self.expand1 = ExpandingBlock(512, 256)
        self.expand2 = ExpandingBlock(256, 128)
        self.expand3 = ExpandingBlock(128, 64)

        # Final convolution layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the U-Net architecture.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after the final convolution.
        """
        # Contracting path
        x, skip1 = self.contract1(x)
        x, skip2 = self.contract2(x)
        x, skip3 = self.contract3(x)
        x, _ = self.contract4(x)

        # Expanding path
        x = self.expand1(x, skip3)
        x = self.expand2(x, skip2)
        x = self.expand3(x, skip1)

        # Final convolution
        x = self.final_conv(x)

        return x

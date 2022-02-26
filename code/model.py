from torch import nn

class ConvBlock(nn.Modeule):

    def __init__(self, in_c, out_c, kernel, padding=None):
        super().__init__()

        self.conv = nn.Conv2d(in_c, out_c, kernel, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((2, 2))

    def forward(self, input):

        x = self.conv(input)
        x = self.relu(x)
        x = self.maxpool(x)

        return x


class DeConvBlock(nn.Module):

    def __init__(self, in_c, out_c, kernel, stride=None):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel, stride=stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):

        x = self.deconv(input)
        x = self.relu(x)

        return x


class IMEncoder(nn.Module):

    def __init__(self, in_c, kernel, padding=None):
        super().__init__()

        self.convB1 = ConvBlock(in_c, 16, kernel, padding)
        self.convB2 = ConvBlock(16, 32, kernel, padding)
        self.convB3 = ConvBlock(32, 64, kernel, padding)
        self.convB4 = ConvBlock(64, 128, kernel, padding)

    def forward(self, input):

        x = self.convB1(input)
        x = self.convB2(x)
        x = self.convB3(x)
        x = self.convB4(x)

        return x


class IMDecoder(nn.Module):

    def __init__(self, out_c, kernel, stride=None):
        super().__init__()

        self.deconvB1 = DeConvBlock(128, 64, kernel, stride)
        self.deconvB2 = DeConvBlock(64, 32, kernel, stride)
        self.deconvB3 = DeConvBlock(32, 16, kernel, stride)
        self.deconvB4 = DeConvBlock(16, out_c, kernel, stride)

    def forward(self, input):

        x = self.deconvB1(input)
        x = self.deconvB2(x)
        x = self.deconvB3(x)
        x = self.deconvB4(x)

        return x

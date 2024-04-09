import torch
import torch.nn as nn
from .blocks import ConvLayer, Decoder

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        ######################################################################################################
        # Part3a Q2 Implement encoder
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        self.conv1 = ConvLayer(6,64,7,2,3) # TODO
        self.conv2 =  ConvLayer(64,128,5,2,2) # TODO
        self.conv3 =  ConvLayer(128,256,5,2,2) # TODO
        self.conv3_1 =  ConvLayer(256,256,3,1,1) # TODO
        self.conv4 =  ConvLayer(256,512,3,2,1) # TODO
        self.conv4_1 =  ConvLayer(512,512,3,1,1) # TODO
        self.conv5 =  ConvLayer(512,512,3,2,1) # TODO
        self.conv5_1 =  ConvLayer(512,512,3,1,1) # TODO
        self.conv6 =  ConvLayer(512,1024,3,2,1) # TODO

        # Note, that the diagram in the paper does not show this layer.
        # However, in the original Caffe code, the authors an additional layer in the bottleneck.
        # This layer Has the same input and output channels as the output of the previous channel and does not downsaple.
        self.conv6_1 =  ConvLayer(1024,1024,3,1,1)  # TODO

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x: torch.Tensor):
        """
        :param x: The two input images concatenated along the channel dimension
        :return: A list of encodings at different stages in the decoder
                As can be seen in the diagram of the FlowNet paper, skip connections branch of at differnt positions.
        """
        ######################################################################################################
        # Part3a Q2 Implement encoder
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        x_1 = self.conv1(x)
        x_2 = self.conv2(x_1)
        x_3 = self.conv3(x_2)
        x_3_1 = self.conv3_1(x_3)
        x_4 = self.conv4(x_3_1)
        x_4_1 = self.conv4_1(x_4)
        x_5 = self.conv5(x_4_1)
        x_5_1 = self.conv5_1(x_5)
        x_6 = self.conv6(x_5_1)
        x_6_1 = self.conv6_1(x_6)

        return  ((x_6_1 , x_5_1 , x_4_1 , x_3_1, x_2))


        pass

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****


class FlowNetS(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, image1: torch.Tensor, image2: torch.Tensor):
        """
        Implement the full forward pass of the FlowNetS model.
        For this, you need to implement both the encoder and decoder.
        All parameters are given in the diagram figure of the FlowNet paper.
        :param image1: First image
        :param image2: Second image
        :return: Flow field
        """

        ######################################################################################################
        # Part3a Q5 Put all components together
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****
        concatenated_image = torch.cat((image1, image2), dim=1)
        t = self.encoder(concatenated_image)
        r =  self.decoder(t)

        return r
        pass

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****


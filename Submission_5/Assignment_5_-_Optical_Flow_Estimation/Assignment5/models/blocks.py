import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat

class ConvLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=3, stride=1, padding=1):
        super().__init__()

        ######################################################################################################
        # Part3a Q1 Implement ConvLayer
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        self.conv =  nn.Conv2d(in_ch, out_ch, kernel_size , stride, padding) # TODO
        self.act =  nn.LeakyReLU(negative_slope=0.1) # TODO Note that we need a LeakyRelu with negative slop parameter 0.1

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of a simple convolution layer by performing the convolution and the activation.
        :param x: Input to the layer
        :return: Output after the activation function
        """

        ######################################################################################################
        # Part3a Q1 Implement ConvLayer
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****
        x= self.conv(x)
        x = self.act(x)
        return x

        pass

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****


class UpConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1, output_padding=0):
        super().__init__()

        ######################################################################################################
        # Part3a Q3 Implement UpconvLayer
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        self.conv =  nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, output_padding, bias= False)# TODO
        self.act = nn.LeakyReLU(negative_slope=0.1) # TODO Note that we need a LeakyRelu with negative slop parameter 0.1
        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of a simple upconvolution layer by performing the convolution and the activation.
        :param x: Input to the layer
        :return: Output of the upconvolution with subsequent activation
        """
        ######################################################################################################
        # Part3a Q3 Implement UpconvLayer
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****
        x= self.conv(x)
        x = self.act(x)
        return x


        pass

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        ######################################################################################################
        # Part3a Q4 Implement decoder
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        self.deconv5 = UpConvLayer(1024,512) # TODO
        self.deconv4 =  UpConvLayer(1026,256)  # TODO
        self.deconv3 =  UpConvLayer(770,128) # TODO
        self.deconv2 =  UpConvLayer(386,64) # TODO

        self.flow5 =  nn.Conv2d(1024, 2 , 3, 1 ,1 , bias=False)
        self.flow4 =  nn.Conv2d(1026, 2 , 3, 1 ,1 , bias= False) # TODO
        self.flow3 = nn.Conv2d(770, 2 , 3, 1 ,1 , bias= False) # TODO
        self.flow2 =  nn.Conv2d(386, 2 , 3, 1 ,1, bias= False ) # TODO
        self.flow1 =  nn.Conv2d(130+64, 2 , 3, 1 ,1 , bias= False) # TODO
        
        
        #F.interpolate(x , scale_factor=4, mode='bilinear', align_corners=False) # TODO

        self.upsample_flow6to5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False) 
        self.upsample_flow5to4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False) 
        self.upsample_flow4to3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False) 
        self.upsample_flow3to2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False) 

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****


    def forward(self, x: list):
        """
        Implement the combined decoder for FlowNetS and FlowNetC.
        :param x: A list that contains the output of the bottleneck, as well as the feature maps of the
          required skip connections
        :return: Final flow field
        Keep in mind that the network outputs a flow field at a quarter of the resolution.
        At the end of the decoding, you need to use bilinear upsampling to obtain a flow field at full scale.
        (hint: you can use F.interpolate).
        """

        ######################################################################################################
        # Part3a Q4 Implement decoder
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        deconv_5 = self.deconv5(x[0])
        flow_5 = self.flow5( x[0])

        deconv_4 = self.deconv4(torch.cat((torch.cat((deconv_5, x[1]), dim=1), self.upsample_flow6to5(flow_5)), dim = 1))
        flow_4 = self.flow4(torch.cat((torch.cat((deconv_5, x[1]), dim=1), self.upsample_flow6to5(flow_5)), dim = 1))


        deconv_3 = self.deconv3(torch.cat((torch.cat((deconv_4, x[2]), dim=1), self.upsample_flow5to4(flow_4)), dim=1))
        flow_3 = self.flow3(torch.cat((torch.cat((deconv_4, x[2]), dim=1), self.upsample_flow5to4(flow_4)), dim=1))

        deconv_2 = self.deconv2(torch.cat((torch.cat((deconv_3, x[3]), dim=1), self.upsample_flow4to3(flow_3)), dim=1))
        flow_2 = self.flow2(torch.cat((torch.cat((deconv_3, x[3]), dim=1), self.upsample_flow4to3(flow_3)), dim=1))

        flow_1 = self.flow1(torch.cat((torch.cat((deconv_2, x[4]), dim=1), self.upsample_flow3to2(flow_2)), dim=1))


        return F.interpolate(flow_1, scale_factor=4, mode='bilinear', align_corners=False)


        pass
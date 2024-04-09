import torch
import torch.nn as nn

from .blocks import ConvLayer, Decoder


class FeatureExtractor(nn.Module):
    def __init__(self):
        """
        Implement the layers in the Siamese feature extractor of the FlowNetC network.
        The specific parameters can be seen in the diagram of the FlowNet paper.
        """
        super().__init__()

        ######################################################################################################
        # Part3b Q1 Implement feature extractor
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        self.conv1 = ConvLayer(3,64,7,2,3)
        self.conv2 = ConvLayer(64,128,5,2,2)# TODO
        self.conv3 = ConvLayer(128,256,5,2,2)# TODO

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x: torch.Tensor):
        """
        :param x: An input image.
        :return: A tuple that returns the output of self.conv2 and self.conv3.
                 In addition to the final output that will be used in the Correlation Layer, we also need the
                 output of self.conv2 as a skip connection to the decoder.
                 Details about this can be found in the diagram of the FlowNet paper.
        """

        ######################################################################################################
        # Part3b Q1 Implement feature extractor
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        conv_1 = self.conv1(x)
        conv_2 = self.conv2(conv_1)
        conv_3 = self.conv3(conv_2)

        return(conv_2,conv_3)

        pass

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

class CorrelationLayer(nn.Module):
    def __init__(self, d=20, s1=1, s2=2):
        super().__init__()
        self.s1 = s1
        self.s2 = s2
        self.d = d
        self.padlayer = nn.ConstantPad2d(d, value=0.0)

    def forward(self, features_1: torch.Tensor, features_2: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the Correlation Layer.

        :param features_1: the feature map obtained from the first image in the sequence
        :param features_2: the feature map obtained from the second image in the sequence
        :return: The correlation of two patches in the corresponding feature maps.
                Use k=0, d = 20, s1 = 1, s2 = 2
        """

        ######################################################################################################
        # Part3b Q2 Implement correlation
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        features_1 = features_1.squeeze()
        features_2 = features_2.squeeze()

        result = torch.zeros((441,48,64))
        padded_features_2 = self.padlayer(features_2)
        
        for i in range(48):
            for j in range(64):
                f1 = features_1[:, i, j]

                # Extract f2 window
                f2 = padded_features_2[:, i:i+2*self.d+1, j:j+2*self.d+1]

                #making dimension of f1 as 256*1*1
                f1 = f1.unsqueeze(1).unsqueeze(2)

                # Reshape f1 for broadcasting f2 new shape 256*21*21
                f1_expanded = f1.expand(-1, 21, 21)

                #dilation factor stride of f2 by 2
                f2_downsampled = f2[:, ::2, ::2]

                # Perform dot product

                result_ = torch.sum(f1_expanded * f2_downsampled , dim=0)
                result [:, i,j ] = result_.flatten()

        return result.unsqueeze(dim = 0)
        pass

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****


class Encoder(nn.Module):

    def __init__(self):
        """
        Implement the Layers of the FlowNetC encoder
        The specific parameters of each layer can be found in the diagram in the paper
        """
        super().__init__()

        ######################################################################################################
        # Part3b Q3 Implement encoder
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        self.conv3_1 = ConvLayer(473,256,3,1,1) # TODO
        self.conv4 = ConvLayer(256,512,3,2,1)# TODO
        self.conv4_1 = ConvLayer(512,512,3,1,1)# TODO
        self.conv5 = ConvLayer(512,512,3,2,1)# TODO
        self.conv5_1 = ConvLayer(512,512,3,1,1)# TODO
        self.conv6 = ConvLayer(512,1024,3,2,1)# TODO

        # Note, that the diagram in the paper does not show this layer.
        # See the comment in the FlowNetS architecture for more details.
        self.conv6_1 = ConvLayer(1024,1024,3,1,1)# TODO

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x: torch.Tensor):
        """
        The forward pass of the FlowNetC encoder
        :param x: The output of the Correlation Layer
        :return: A List of encodings at different stages in the decoder.
                As can be seen in the diagram of the FlowNet paper, skip connections branch at different positions
                in the encoder into the decoder.
        """

        ######################################################################################################
        # Part3b Q3 Implement encoder
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        x_3_1 = self.conv3_1(x)
        x_4 = self.conv4(x_3_1)
        x_4_1 = self.conv4_1(x_4)
        x_5 = self.conv5(x_4_1)
        x_5_1 = self.conv5_1(x_5)
        x_6 = self.conv6(x_5_1)
        x_6_1 = self.conv6_1(x_6)

        return  ((x_6_1 , x_5_1 , x_4_1 , x_3_1))
        pass

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

class FlowNetC(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.conv_redir =  ConvLayer(256,32,1,1, 0)# TODO
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.relu = nn.ReLU()
        self.correlationLayer = CorrelationLayer()

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward pass of the FlowNetC model.
        :param image1: First image
        :param image2: Second image
        :return: Flow field
        """

        ######################################################################################################
        # Part3b Q4 Put components together
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        # Encode both images using the Feature extractor
        img_1_c2 ,img_1_c3 = self.feature_extractor(image1)
        img_2_c2 ,img_2_c3 = self.feature_extractor(image2)

        # Establish correlation volume
        out_correlation = self.correlationLayer(img_1_c3, img_2_c3)

        # Use the ReLU on the correlation layer

        out_correlation = self.relu(out_correlation)

        out_conv_redir = self.conv_redir(img_1_c3)


        enc_inp = torch.cat([out_conv_redir, out_correlation], dim=1)
        op = self.encoder(enc_inp)

        # Attach the decoder

        op = self.decoder(op + (img_1_c2,))

        return op

        pass

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

    def correlate(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """
        PLEASE SEE EXCERCISE SHEET FOR INFORMATION.
        :param image1: The first image in the image sequence as a torch tensor
        :param image2: The second image in the image sequence as a torch tensor
        :return: The output of the correlation layer after encoding the images
        The goal of is function is to give you an additional opportunity to debug the Correlation layer
        This function will be called in the test_correlation function in the ModelWrapper
        """

        ######################################################################################################
        # Part3b Q4 Put components together
        ######################################################################################################
        # ***** START OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

        # Extract features from both images
        img_1_c2 ,img_1_c3 = self.feature_extractor(image1)
        img_2_c2 ,img_2_c3 = self.feature_extractor(image2)


        # Compute the correlation volume (shape should be [1, 441, 48, 64])

        out_correlation = self.correlationLayer(img_1_c3, img_2_c3)
        return out_correlation

        pass

        # ***** END OF YOUR ANSWER (DO NOT DELETE/MODIFY THIS LINE)*****

import torch
from tqdm import tqdm

def get_conv(output_size, input_shape):

    input_size = input_shape[-2:]
    in_channels, out_channels = input_shape[1], input_shape[1]

    # Determine: Kernel Size

    kx = (input_size[0] + 1) - output_size[0]
    ky = (input_size[1] + 1) - output_size[1]
    kernel_size = (ky,kx)

    # Initialize: Convolutional Layer

    layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1)

    return layer

def get_conv_transpose(input_size, in_channels, out_channels, mod_size):

    #Initialize the output list

    output_size = [0,0]    

    #Split the input size

    #input_size = (sample.shape[-1], sample.shape[-2])

    #Get the next largets output size for the spatial dimensions given the mod size
    for i in range(input_size[0], input_size[0] + mod_size):
        output_size[0] = i
        if output_size[0] % mod_size == 0 : break
    for i in range(input_size[1], input_size[1] + mod_size):
        output_size[1] = i
        if output_size[1] % mod_size == 0 : break

    #Determine the kernel
    kx = output_size[0] - (input_size[0] - 1)
    ky = output_size[1] - (input_size[1] - 1)
    kernel_size = (ky,kx)
    
    #Initialize the layer
    layer = torch.nn.ConvTranspose2d(in_channels, 
                                     out_channels, 
                                     kernel_size, 
                                     stride=1, 
                                    )
    return layer


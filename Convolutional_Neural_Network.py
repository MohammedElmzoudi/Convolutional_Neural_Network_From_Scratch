import numpy as np

class OCR:
    ## CNN Architecture
        # nxn size input image
        # 6 - 5x5 - stride 1 - kernels
        # 2x2 max pooling layer
        # 16 - 5x5 - stride 1 - kernels
        # 2x2 max pooling layer
        # Fully Connected Layer Nx1
        # Fully Connected Layer 120x1
        # Fully Connected Layer 84x1
    
    def __init__(self):
        #Define the previous layer computed - allows for modularity of the model
        #After each computed layer, update previous layer array and dimensions 
        self.previous_layer = np.zeros([1,255,255])
        self.pl_dims = np.array(self.previous_layer.shape[1:])
        self.image_dims = np.array([25,25])

        #Load weight values here
        self.kernels1 = np.zeros([6,5,5])
        self.kernels2 = np.zeros([16,5,5])

    #Convolves kernels with data in the previous layer
    #Outputs a tensor with feature maps for each kernel
    def cnn_layer(self,kernels):
        #Kernel tensor dimensions = (#kernels,l,w)
        num_kernels = kernels.shape[0]
        kernel_dims = np.array(kernels.shape[1:])
        
        #Using full padding, to not have overlap in convolution
        target_dims = self.pl_dims + kernel_dims - 1
        output = np.zeros([num_kernels,target_dims[0],target_dims[1]])

        #Takes the sum of convolution between each image in the previous layer and kernel, for each kernel
        for k in range(num_kernels):
            for o in range(self.previous_layer.shape[0])
                output[k] += np.fft.fft2(self.previous_layer[o],target_dims) * np.fft.fft2(kernels[k],target_dims)
                
            
        self.previous_layer = output
        self.pl_dims = np.array(output.shape[1:])
        

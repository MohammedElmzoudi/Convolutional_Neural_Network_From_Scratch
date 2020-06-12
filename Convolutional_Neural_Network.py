import numpy as np
from scipy.special import softmax, expit

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
        self.previous_layer = np.zeros([1,32,32])
        self.pl_dims = np.array(self.previous_layer.shape[1:])
        self.image_dims = np.array([25,25])
        self.pool_size = 2
        
        #Load weight values here
        self.kernels1 = np.random.rand(6,5,5)
        self.kernels2 = np.random.rand(16,5,5)
        self.kernels3 = np.random.rand(120,5,5)

        self.weights1 = np.random.rand(84,120)
        self.weights2 = np.random.rand(10,84)

    #Convolves kernels with data in the previous layer
    #Outputs a tensor with feature maps for each kernel
    def cnn_layer(self, kernels, activation = 'tanh', inp = None):

        if inp is None:
            inp = self.previous_layer
        #Kernel tensor dimensions = (#kernels,l,w)
        inp_dims = np.array(inp.shape)[1:]
        num_kernels = kernels.shape[0]
        kernel_dims = np.array(kernels.shape[1:])
        
        #Using full padding, to not have overlap in convolution
        target_dims = inp_dims + kernel_dims - 1
        output = np.zeros([num_kernels,target_dims[0],target_dims[1]])

        #Takes the sum of convolution between each image in the previous layer and kernel, for each kernel
        for k in range(num_kernels):
            for o in range(inp.shape[0]):
                fft_ar = np.fft.fft2(inp[o],target_dims) * np.fft.fft2(kernels[k],target_dims)
                target = np.fft.ifft2(fft_ar).real
                output[k] = output[k] + target

        #output = self.activation(output,activation)
        self.previous_layer = output
        self.pl_dims = np.array(output.shape[1:])
        return output
        
    def max_pooling(self, features = None, stride = 2, size = np.array([2,2])):
        if features is None:
            features = self.previous_layer
        feature_dims = np.array(features.shape[1:])
        target_dims = np.uint16(((feature_dims - size)/stride)+1)
        pooled_features = np.zeros([features.shape[0],target_dims[0],target_dims[1]])
        for nF in range(features.shape[0]):
            
            for r in range(pooled_features.shape[1]):
                r_start = r * stride
                r_start = np.clip(r_start, 0, feature_dims[0]-1)
                r_end = r_start + size[0]
                r_end = np.clip(r_end,0,feature_dims[0])
                
                for c in range(pooled_features.shape[2]):
                    c_start = c * stride
                    c_start = np.clip(c_start, 0, feature_dims[1]-1)
                    c_end = c_start + size[0]
                    c_end = np.clip(c_end,0,feature_dims[1])

                    patch = features[nF, r_start:r_end, c_start:c_end ]
                    pooled_features[nF,r,c] = np.max(patch)
                
        self.previous_layer = pooled_features
        self.pl_dims = np.array(pooled_features.shape[1:])
        return pooled_features

    def flatten_layer(self, inp = None):
        if inp is None:
            inp = self.previous_layer
        output = inp.flatten()
        output = np.expand_dims(output,axis = 1)
        self.previous_layer = output
        return output

    def fully_connected_layer(self,weights, inp = None, activation = 'softmax'):
        if inp is None:
            inp = self.previous_layer
        output = np.matmul(weights,inp)
        output = self.activation(output,activation)
        self.previous_layer = output

        return output
    def activation(self, inp, activation):

        output = None
        if activation is 'tanh':
            output = np.tanh(inp)

        if activation is 'softmax':
            output = softmax(inp)

        if activation is 'sigmoid':
            output = expit(inp)

        if activation is 'relu':
            output = inp[inp<0] = 0
        return output


model = OCR()
print('------------Model_Summary------------')

model.cnn_layer(kernels = model.kernels1)
print("After Convolution 1: ", model.previous_layer.shape)
model.max_pooling()
print("After Pooling 1: ", model.previous_layer.shape)

model.cnn_layer(kernels = model.kernels2)
print("After Convolution 2: ", model.previous_layer.shape)
model.max_pooling()
print("After Pooling 2: ", model.previous_layer.shape)

model.cnn_layer(kernels = model.kernels3)
print("After Convolution 3: ", model.previous_layer.shape)

model.flatten_layer()
print("Flattened: ",model.previous_layer.shape)

model.fully_connected_layer(weights = model.weights1)
print("After Fully Connected 1: ", model.previous_layer.shape)

model.fully_connected_layer(weights = model.weights2)
print("After Fully Connected 2: ", model.previous_layer.shape)

















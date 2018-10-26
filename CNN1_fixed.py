import numpy as np 
from ActivationFunction import ActivationFunction as af


class CNN(object):
    
    
    def __init__(self, definition):
        self.definition = definition
        self.layers = self.read_definition(definition)


    #function that creates a filter of depth [d], height [h], and width [w]
    
    def gen_filter (self, d, h, w):
        return np.random.random((d, h, w))-0.5

    #function that reads a definition of a CNN, and generates a list of filters/weights
    def read_definition(self, l):
        layerCNT = l[0]
        layers =[]
        #looking through each layer 
        for i in range (3, layerCNT+3):
            (filterCnt, (h, w), _) = l[i]
            d         = l[i-1][0]
            filters   = []
            #creating the filters for a layer
            for j in range (0, filterCnt):
                filters.append(self.gen_filter (d, h, w))  
            # add  [filters] to the layers list 
            layers.append(filters)
        cs = self.conv_size(l)
        (fD, (fH, fW)) = cs[len(cs)-1]
        fcc = []
        for k in range (0, l[1]):
            fcc.append(self.gen_filter(fD, fH, fW))
        layers.append(fcc)
        return np.array(layers)


    def gen_filter_bp (self, d, h, w):
        return np.zeros((d, h, w))

    #function that reads a definition of a CNN, and generates a list of filters/weights
    def read_definition_bp(self, l):
        layerCNT = l[0]
        layers =[]
        #looking through each layer 
        for i in range (3, layerCNT+3):
            (filterCnt, (h, w), _) = l[i]
            d         = l[i-1][0]
            filters   = []
            #creating the filters for a layer
            for j in range (0, filterCnt):
                filters.append(self.gen_filter_bp (d, h, w))  
            # add  [filters] to the layers list 
            layers.append(filters)
        cs = self.conv_size(l)
        (fD, (fH, fW)) = cs[len(cs)-1]
        fcc = []
        for k in range (0, l[1]):
            fcc.append(self.gen_filter_bp(fD, fH, fW))
        layers.append(fcc)
        return np.array(layers)

    #function that creates a list of output dimentions after each layer in the cnn
    # [l] is the list representation of cnn architecture 
    def conv_size(self, l):
        layerCnt = l[0]
        
        cSize = [[l[2][0], l[2][1]]]
        for i in range (3, layerCnt+3):
            (d, (h, w), s) = l[i]
            (in_d, (in_h, in_w)) = cSize[i-3]
        
            #calculate new dimensions
            w_new = int((in_w - w)/s + 1)
            h_new = int((in_h - h)/s + 1) 
            d_new = d
            
            cSize.append([d_new, [h_new, w_new]])
        return cSize

    #a function that completes one forward pass through the network on [image]
    # returns logit with sofmax applied
    #image is 3d array 
    #[l] is the network architecture in list form, product of red_definition(smth)
    def forward_pass(self, image):
        defi = self.definition
        layers = self.layers
        
        output = self.conv_size(defi)
        layerCnt = len(layers)
        im = image
        hidden = [im]
        
        #to go through each layer of cnn
        for l in range (0, layerCnt-1):
            #information about input
            (imageD, imageH, imageW) = np.shape(im)
            
            #information about filtering 
            (_, (fH, fW), stride) = defi[l+3]
             
            #information about output
            (outD, (outH, outW)) = output[l+1]
            fltrCnt = outD
            
            #create empty array to fill, represents pass of current layer
            out = np.zeros((outD, outH, outW))
            
            #filters for layer 
            fltr = layers[l]
            #
            #convolute and apply filter
            for hIdx in range (0, outH):
                h_ = hIdx*stride
                #
                for wIdx in range (0, outW):
                    w_ = wIdx*stride
                    sub = im[:, h_:h_+fH, w_:w_+fW]
                    #
                    # iterate through each filter
                    for fIdx in range (0, fltrCnt): 
                        fWt = fltr[fIdx]
                        out[fIdx][hIdx][wIdx] +=  af.Leaky_ReLU(np.sum(((np.multiply(sub, fWt))))) 
                       # out[fIdx][hIdx][wIdx] +=  np.sum(((np.multiply(sub, fWt)))) 
            #
            hidden.append(out)
            im = out
            #im = af.Leaky_ReLU(out)
        #
        #   
        #fully connected layer
        weights = layers[len(layers)-1]
        dot = []
        #for w in range (0, len(weights)):
        for wt in weights:
            dot.append(np.sum(np.multiply(im, wt)))
        #
        #hidden.append(np.array(dot))
        logit = af.Leaky_ReLU(np.array(dot))
        hidden.append(logit)
        #
        #hidden layers from conv, the logit, the softmax
        return np.array(hidden), logit, np.array(af.softmax(logit, shift=False))


    ######################## back propagation helpers ############################
    #
    # back propagation - loss function
    # uses cross entropy 
    # [target] is the larget vector
    # [actual] is the network output, after softmax
    # returns gradient , 1Xn
    def loss_function_bp(self, target, softMax ):
        crossEntrp =  -1*(target/softMax)
        return crossEntrp

    # [lf_grad]
    def soft_max_bp(self, soft ): #nXn is soft max 
        sm = af.softmax_deriv(soft)
        return sm 

    def relu_bp (self, dot): #diagonal matrix 
        rl = af.Leaky_ReLU_deriv(dot)
        return rl


    # a function that completes one backwards pass through the network on [image]
    # produces a list of matrixes with adjustment values 
    # target is the target vector
    #[l] is the network architecture in list form, product of read_definition(smth)
    #[h] hidden layers
    # [softMax] is the logit after the soft max
    def backward_propegation (self, target, softMax, hidden, logit):
        defi = self.definition
        layers = self.layers
        
        layerDi = self.conv_size(defi)
        layerCnt = len(layerDi)
        adj_matrix = self.read_definition_bp(defi)
        
        loss_gd = self.loss_function_bp(target, softMax)
        soft_gd = np.matmul(loss_gd, self.soft_max_bp(softMax))
        rel_gd = np.multiply(soft_gd, self.relu_bp((hidden[-1])))
        #
        #fully connected layer-----------------------------------------------
        inputs_ff = hidden[len(hidden) -2]
        for nCnt in range(0, len(rel_gd)):
            adj_matrix[-1][nCnt] = np.multiply(rel_gd[nCnt], inputs_ff)
        #
        #go through any number of connvolutional layers----------------------
        prev_gd = np.zeros(np.shape(layers[-1][0]))
        #
        for i in range(0, len(rel_gd)):
            prev_gd += rel_gd[i]*layers[-1][i]
        #
        for l in range(layerCnt-1, 0, -1): 
            #first entry in the hidden is the input dimentions
            output = hidden[l]
            #input
            input = hidden[l-1]
            #
            #dimension of the input 
            (inD, (inH, inW)) = layerDi[l-1] 
            #dimensions of the filter
            (_, (fH, fW), stride) = defi[l+2]
            fD = inD
            (fCnt, _, _, _) = np.shape(layers[l-1])
            #dimension of the output
            (outD, (outH, outW)) = layerDi[l]
            #
            #
            #filter_gd = np.zeros(np.shape(adj_matrix[l-1]))
            #
            #relu derivativenp
            reluCL_gd = af.Leaky_ReLU_deriv(output)
            #
            prev_gd = prev_gd * reluCL_gd
            #
            #go through entries in the input
            for hInx in range(0, outH):
                h_ = hInx*stride
                #
                for wInx in range(0, outW):
                    w_ = wInx*stride
                    #
                    #sub section of input that corresponds to current filter
                    inSub = input[:, h_:h_+fH, w_:w_+fW] 
                    #
                    for fCnt in range (0, outD):
                        #the dos produce of filter and sub section
                        out_gd = prev_gd[fCnt][hInx][wInx]
                        #out_gd = out_gd * reluCL_gd[fCnt][hInx][wInx]
                        #
                        adj_matrix[l-1][fCnt] += np.multiply(out_gd, inSub)      
            #
            #
            #multiply gradients from the dot product and gradients from the relu
            #fill adj_matrix with the product
            '''
            for d in range(0, outD):
                    for w in range(0, outW):
                        adj_matrix[l-1][d] += prev_gd[d][h][w] * filter_gd[d]
            '''
            #
            #adjust the gradient for the next layer
            intra_gd = np.zeros((inD, inH, inW))
            #
            for d in range(0, outD):
                fCurrent = layers[l-1][d]
                #
                for h in range(0, outH):
                    hStart = h*stride
                    #
                    for w in range(0, outW):
                        wStart = w*stride
                        #
                        intra_gd[:, hStart:hStart+fH, wStart:wStart+fW] += (prev_gd[d][h][w] * fCurrent)
            #
            prev_gd = intra_gd
        #
        return adj_matrix
    
    
    def loss_fx (self, target, softMax):
        loss = np.sum([0 if target[k] == 0 else -target[k]*np.log(softMax[k]/target[k]) for k in range(0, len(target))])
        return loss
    
    
    def check_back_prop (self, image, defi, target, lry):
        #
        adj_amt = 0.000001
        #
        #[layers] are the weights 
        layers = self.layers
        #
        hidden, logit, softMax = self.forward_pass(image)
        #
        loss = self.loss_fx(target, softMax) #calculate the loss function value 
        #
        adj = self.backward_propegation(target, softMax, hidden, logit)
        #
        (f, d, h, w) = np.shape(adj[lry])
        (F, D, H, W) = (np.random.randint(0,f), np.random.randint(0,d), np.random.randint(0,h), np.random.randint(0,w) )
        #
        grd = adj[lry][F][D, H, W]
        #
        #change the filter that you are looking at 
        layers[lry][F][D, H, W] += adj_amt
        #
        #excepted loss value change
        p_lossAdj = grd * adj_amt
        #
        #redo the forward propegation
        _, _, softNew = self.forward_pass(image)
        #
        #find the loss
        lossNew = self.loss_fx(target, softNew)
        #
        #layers[lry][0][0, 0, 0] -= adj_amt
        #
        loss_change = lossNew - loss
        #
        print("Original Loss=", loss, "\nNew Loss: ", lossNew , "\nChange in loss =   ", loss_change, "\nPredicted Change = " , p_lossAdj, "Weight Gradient", grd)





    

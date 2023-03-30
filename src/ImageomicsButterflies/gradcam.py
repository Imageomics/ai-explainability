
from PIL import Image
from matplotlib import transforms

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, ToPILImage

class Embed(nn.Module):
    def __init__(self):
        super(Embed,self).__init__()
        
        self.conv1 = nn.Conv2d(3,32,9,padding=4)
        torch.nn.init.orthogonal(self.conv1.weight)

        self.p1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32,96,3,padding=1)
        torch.nn.init.orthogonal(self.conv2.weight)
        self.conv2b = nn.Conv2d(96,48,1)
        torch.nn.init.orthogonal(self.conv2b.weight)
        self.p2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(48,128,3,padding=1)
        torch.nn.init.orthogonal(self.conv3.weight)
        self.conv3b = nn.Conv2d(128,64,1)
        torch.nn.init.orthogonal(self.conv3b.weight)        
        self.p3 = nn.MaxPool2d(2)
        
        self.conv4 = nn.Conv2d(64,96,3,padding=1)
        torch.nn.init.orthogonal(self.conv4.weight)        
        self.p4 = nn.MaxPool2d(2)
        
        self.conv5 = nn.Conv2d(96,128,3,padding=1)
        torch.nn.init.orthogonal(self.conv5.weight)        
        self.conv6 = nn.Conv2d(128,128,3,padding=1)
        torch.nn.init.orthogonal(self.conv6.weight)        
        self.p5 = nn.MaxPool2d(2)
        
        self.dense1 = nn.Linear(10*128,256)
        self.dense2 = nn.Linear(256,256)
        self.dense3 = nn.Linear(256,64)

        self.target_layer_output = None
        self.target_gradients = None

    def save_gradients(self, grad):
        self.target_gradients = grad
        
    def inter(self, x):
        z = self.p1(F.elu(self.conv1(x)))
        #z = self.p2(F.elu(self.conv2b(F.elu(self.conv2(z)))))
        #z = self.p3(F.elu(self.conv3b(F.elu(self.conv3(z)))))
        #z = self.p4(F.elu(self.conv4(z)))
        
        #z = F.elu(self.conv5(z))
        #z = z + F.elu(self.conv6(z))
        
        return z

    def forward(self, x):
        z = self.p1(F.elu(self.conv1(x)))
        z = self.p2(F.elu(self.conv2b(F.elu(self.conv2(z)))))
        z = self.p3(F.elu(self.conv3b(F.elu(self.conv3(z)))))
        z = self.p4(F.elu(self.conv4(z)))
        
        z = F.elu(self.conv5(z))
        # ! =======================
        # Notice how I picked this conv6 layer as my target layer
        # If you want to change the layer, you have to go to that layer can get the output 'z', register this hook, and save z to self.target_layer_output
        z = self.conv6(z)
        z.register_hook(self.save_gradients)
        self.target_layer_output = z
        # ! =======================
        z = z + F.elu(self.conv6(z))
        z = self.p5(z)
        
        s = z.size()
        z = z.permute(0,1,2,3).contiguous().view(s[0],s[1]*s[2]*s[3])
        print(z.shape)
        z = F.elu(self.dense1(z))
        z = F.elu(self.dense2(z))
        z = self.dense3(z)
        
        z = z / torch.sqrt(torch.sum(z**2,1,keepdim=True)+1e-16)
        
        return z

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.embed = Embed()
        self.dense1 = nn.Linear(64,96)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(96,96)
        self.drop2 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(96,96)
        self.drop3 = nn.Dropout(0.5)
        self.dense4 = nn.Linear(96,12)
        
        self.adam = torch.optim.Adam(self.parameters(), lr=1e-4)

    def get_embedding(self,x):
        return self.embed(x)
    
    def predict(self, z):
        z = self.drop1(F.elu(self.dense1(z)))
        z = self.drop2(F.elu(self.dense2(z)))
        z = self.drop3(F.elu(self.dense3(z)))
        p = F.softmax(self.dense4(z),dim=1)
        
        return p
        
    def forward(self,x1,x2,x3):
        z1 = self.get_embedding(x1)
        z2 = self.get_embedding(x2)
        z3 = self.get_embedding(x3)
        
        p = self.predict(z1)
        
        return p,z1,z2,z3    

    def triplet(self,z1,z2,z3):
        return torch.mean((z1-z2)**2) - torch.mean((z1-z3)**2)
    
    def loss(self,p,z1,z2,z3,y):
        tloss = self.triplet(z1,z2,z3)
        
        p = torch.clamp(p,1e-6,1-1e-6)
        idx = torch.LongTensor(np.arange(p.size()[0])).cuda()        
        loss = torch.mean(-torch.log(p[idx,y[idx]]))
        
        reg = 0
        for param in self.parameters():
            reg = reg + torch.mean(param**2)
            
        loss = loss + 6e-5*reg + 0.1 * tloss
        
        return loss

input_image = torch.ones(1, 3, 64, 160).cuda() # TODO: Change this to desired preprocessed image
net = Net().cuda()
net.eval()
# TODO: set target class here
target_class = None
# TODO: load pretrained model here
out = net.predict(net.get_embedding(input_image))
if target_class is None:
    target_class = np.argmax(out.detach().cpu().numpy())
# Target for backprop
one_hot_output = torch.FloatTensor(1, out.size()[-1]).zero_()
one_hot_output[0][target_class] = 1
one_hot_output = one_hot_output.cuda()
# Zero grads
net.embed.zero_grad()
net.dense1.zero_grad()
net.dense2.zero_grad()
net.dense3.zero_grad()
net.dense4.zero_grad()
# Backward pass with specified target
out.backward(gradient=one_hot_output, retain_graph=True)
# Get hooked gradients
guided_gradients = net.embed.target_gradients.detach().cpu().numpy()[0]
# Get convolution outputs
target = net.embed.target_layer_output.detach().cpu().numpy()[0]
print(target.shape)
# Get weights from gradients
weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
# Create empty numpy array for cam
cam = np.ones(target.shape[1:], dtype=np.float32)
# Have a look at issue #11 to check why the above is np.ones and not np.zeros
# Multiply each weight with its conv output and then, sum
for i, w in enumerate(weights):
    cam += w * target[i, :, :]
cam = np.maximum(cam, 0) # Relu

# Visualization
cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize

heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
cam = cv2.resize(cam, (64, 160))
heatmap = cv2.resize(heatmap, (64, 160))
raw_input_image = np.ones((3, 64, 160), dtype=np.uint8) * 255 # TODO: Load the image without the transformation, but as the shape 3, 64, 160
raw_input_image = np.swapaxes(raw_input_image, 0, 2)
superimposed = cv2.addWeighted(heatmap, 0.7, cv2.cvtColor(raw_input_image, cv2.COLOR_RGB2BGR), 0.3, 0)

cam = Image.fromarray(cam)
heatmap = Image.fromarray(heatmap)
superimposed = Image.fromarray(superimposed)

cam.save("cam.png")
heatmap.save("cam_heatmap.png")
superimposed.save("cam_superimposed.png")
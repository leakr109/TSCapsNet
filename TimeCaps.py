import torch
import torch.nn as nn
import pytorch_lightning as pl

'''
    TO DO:
    - cell a conv2d kernel/stride
    - add Ln
'''

def squash(s):
    sq_norm = (s**2).sum(-1, keepdim=True)
    v = (sq_norm / (1 + sq_norm)) * (s / torch.sqrt(sq_norm + 1e-8))
    return v
    

class Cell_A(nn.Module):
    def __init__(self, k, cp, ap, g2, cSA, aSA, g3):
        super().__init__()
        self.L = L
        self.k, self.g1 = k, g1
        self.cp, self.ap, self.g2 = cp, ap, g2
        self.cSA, self.aSA, self.g3 = cSA, aSA, g3

        self.conv1 = nn.Conv1d(k, self.cp * self.ap, kernel_size=g2, stride=1, padding = 'same')  # ask
        self.conv2 = nn.Conv2d(1, self.cSA * self.aSA, kernel_size=(self.g3, self.ap), stride=(1, self.ap), padding = 'same')
        

    def forward(self, x):
        x = self.conv1(x)                                              #[L, cp*ap]
        x = x.view(x.size(0), x.size(1), self.cp, self.ap)             #[L, cp, ap]
        x = squash(x)
        x = x.view(x.size(0), x.size(1), self.cp * self.ap, 1)         #[L, cp*ap, 1]
        x = self.conv2(x)                                              #[L, cp, cSA*aSA]
        x = x.view(x.size(0), x.size(1), self.cp, self.cSA * self.aSA) #[L, cp, cSA, aSA]
        #ROUTING                                                       #[L, cSA, aSA]
        #flatten                                                       #[L*cSA, aSA]
        return x
        
    

class Cell_B(nn.Module):
    def __init__(self, cb, ab, g2, g3, Ln, n, cSB, aSB):
        super().__init__()
        k=self.k
        self.cb, self.ab = cb, ab
        self.g2, self.g3 = g2, g3
        self.Ln, self.n = Ln, n
        self.cSB, self.cSA = cSB, aSB

        self.convLayer = nn.Conv1d(k, self.cb, kernel_size=1, padding = 'same')  #ask
        
        self.conv1 = nn.Conv1d(
            self.cb, self.cb * self.ab, 
            kernel_size=g2, 
            stride=1, 
            padding = 'same')
        
        self.conv2 = nn.Conv2d(
            1, self.cSB * self.aSB, 
            kernel_size=(self.g3, self.cb * self.ab), 
            stride=(cb * ab, 1), 
            padding = 'same')

    def forward(delf, x):                                               #[L, k]
        x = self.convLayer(x)                                           #[L, cb]
        x = self.conv1(x)                                               #[L, cb*ab]
        x = x.view(x.size(0), self.Ln, self.n, self.cb * self.ab)       #[l/n, n, cb*ab]
        x = squash(x)
        x = x.view(x.size(0), self.Ln, self.n * self.cb * self.ab, 1)   #[L/n. n*cb*ab, 1]
        x = self.conv2(x)                                               #[L/n, n, cSB*aSB]
        x = x.view(x.size(0), self.Ln, self.cSB, self.aSB, self.n)      #[L/n, cSB, aSB, n]
        #ROUTING
        #flatten



class TimeCaps(pl.LightningModule):
    def __init__(self, L, k, g1, ...):
        super().__init__()
        self.L = L
        self.k, self.g1 = k, g1
        self.cp, self.ap, self.g2 = cp, ap, g2
        self.cSA, self.aSA, self.g3 = cSA, aSA, g3
        self.cb, self.ab = cb, ab

        self.conv1 = nn.conv1d(1, k, kernel_size=g1, stride=1, padding = 'same')
        
        self.cell_A = Cell_A(
            k=self.k, cp=self.cp, ap=self.ap, g2=self.g2, cSA=self.cSA, aSA=self.aSA, g3=self.g3
        )
        self.cell_B = Cell_B(
            k=self.k, cb=self.cb, ab=self.ab, g2=self.g2, g3=self.g3, Ln=self.Ln, n=self.n, cSB=self.cSB, aSB=self.aSB, gB=self.gB
        )

    
    def forward(self, x):
        X = self.conv1(x)
        X_A = self.cell_A(X)
        X_B = self.cell_B(X)
        #x = concat(X_A, X_B)
        #...


import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


def squash(s):
    sq_norm = (s**2).sum(-1, keepdim=True)
    v = (sq_norm / (1 + sq_norm)) * (s / torch.sqrt(sq_norm + 1e-8))
    return v


def routing(U, routingIterations):
    batch, d1, d2, d3, d4 = U.shape
    b = torch.zeros(batch, d1, d2, d3, device=U.device)
    
    for r in range(routingIterations):
        c = F.softmax(b, dim=-1)   #(b, d1, d2, d3)

        c = c.unsqueeze(-1)       #(b, d1, d2, d3, 1)
        s = (c * U).sum(dim=2)
        v = squash(s)

        a = (U * v.unsqueeze(1)).sum(-1)  # agreement
        b = b + a.detach()
        
    return v
        


class Cell_A(nn.Module):
    def __init__(self, k, cp, ap, g2, cSA, aSA, g3):
        super().__init__()
        self.L = L
        self.k, self.g1 = k, g1
        self.cp, self.ap, self.g2 = cp, ap, g2
        self.cSA, self.aSA, self.g3 = cSA, aSA, g3

        self.conv1 = nn.Conv1d(
            k, self.cp * self.ap,
            kernel_size=g2, 
            stride=1, 
            padding = 'same') 
        
        self.conv2 = nn.Conv2d(
            1, self.cSA * self.aSA, 
            kernel_size=(self.g3, self.ap), 
            stride=(1, self.ap), 
            padding = 'same')
        

    def forward(self, x):                                              #[k, L] -
        x = self.conv1(x)                                              #[cp*ap, L] -
        # fix this (solves the problem)                                #[L, cp*ap] -> [L, cp, ap]
        x = squash(x)
        
        x = x.view(x.size(0), self.L, self.cp * self.ap, 1)            #[L, cp*ap, 1]
        x = self.conv2(x)                                              #[L, cp, cSA*aSA]
        
        x = x.view(x.size(0), x.size(1), self.cp, self.cSA * self.aSA)  #[L, cp, cSA, aSA]
        x = routing(x, routIter)                                        #[L, cSA, aSA]
        x = x.view(x.size(0), self.L * self.cSA, self.aSA)              #[L*cSA, aSA]
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
            padding = 'same')  #fix

    def forward(delf, x):                                               #[k, L] -
        x = self.convLayer(x)                                           #[cb, L] -
        
        x = self.conv1(x)                                               #[cb*ab, L] -
        x = x.view(x.size(0), self.Ln, self.n, self.cb * self.ab)       #[l/n, n, cb*ab] #fix
        x = squash(x)  #squash along the full set of feature maps
        
        x = x.view(x.size(0), self.Ln, self.n * self.cb * self.ab, 1)   #[L/n. n*cb*ab, 1]
        x = self.conv2(x)                                               #[L/n, n, cSB*aSB]
        x = x.view(x.size(0), self.Ln, self.cSB, self.aSB, self.n)      #[L/n, cSB, aSB, n] ?
        x = routing(x, routIter)                                        #[L/n, cSB, aSB]
        x = x.view(x.size(0), self.Ln * self.cSB, self.aSB)             #[l/n*cSb, aSB]
        return x



class TimeCaps(pl.LightningModule):
    def __init__(self, L, k, g1, ...):
        super().__init__()
        self.L = L
        self.k, self.g1 = k, g1
        self.cp, self.ap, self.g2 = cp, ap, g2
        self.cSA, self.aSA, self.g3 = cSA, aSA, g3
        self.cb, self.ab = cb, ab

        assert L % n == 0,

        self.conv1 = nn.conv1d(1, k, kernel_size=g1, stride=1, padding = 'same')
        
        self.cell_A = Cell_A(
            k=self.k, cp=self.cp, ap=self.ap, g2=self.g2, cSA=self.cSA, aSA=self.aSA, g3=self.g3
        )
        self.cell_B = Cell_B(
            k=self.k, cb=self.cb, ab=self.ab, g2=self.g2, g3=self.g3, Ln=self.Ln, n=self.n, cSB=self.cSB, aSB=self.aSB, gB=self.gB
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
 

    
    def forward(self, x):
        X = self.conv1(x)   #[k,L] -
        X_A = self.cell_A(X)
        X_B = self.cell_B(X)
        #x = concat(X_A, X_B)
        return x


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5) 

    '''
    def concat(cell_A, cell_B):
        return torch.cat(
    '''

    # margin loss
    def lossF(self, out, y, m_plus=0.9, m_minus=0.1, λ=0.5):
        y_onehot = F.one_hot(y, num_classes=out.size(1)).float()
        norm = torch.norm(out, dim=-1)
        loss = y_onehot * F.relu(m_plus - norm)**2 + λ * (1 - y_onehot) * F.relu(norm - m_minus)**2
        return loss.sum(dim=1).mean()

    
# ------------------- Train -------------------------------------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        out = self(x)
        
        loss = self.lossF(out, y)
        self.log("train_loss", loss)

        preds = torch.norm(out, dim=-1).argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("train_acc", acc)

        return loss   
        
    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics["train_loss"].item()
        train_acc = self.trainer.callback_metrics["train_acc"].item()
        print(f"Epoch: {self.current_epoch}, Loss: {avg_loss:.4f}, Accuracy: {train_acc*100:.2f}%")



    


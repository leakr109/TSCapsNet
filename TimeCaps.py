import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim


def squash(s):
    sq_norm = (s**2).sum(-1, keepdim=True)
    v = (sq_norm / (1 + sq_norm)) * (s / torch.sqrt(sq_norm + 1e-8))
    return v


def routing(V, routingIterations):
    dims = V.shape    #(b, d1, d2, d3, d4)
    b = torch.zeros(dims[:-1], device=V.device)
    
    for r in range(routingIterations):
        c = F.softmax(b, dim=-1)   #(b, d1, d2, d3) -

        c = c.unsqueeze(-1)       #(b, d1, d2, d3, 1) * V ,skalar se pomnoži z vektorji
        s = (c * V).sum(dim=-3)    #(b, d1, d3, d4)
        v = squash(s)

        if (r < routingIterations - 1):
            a = (V * v.unsqueeze(-3)).sum(-1)  # agreement  (b, d1, d2, d3)
            b = b + a.detach()
        
    return v



class Cell_A(nn.Module):
    def __init__(self, L, k, cp, ap, g2, cSA, aSA, g3, routIter):
        super().__init__()
        self.L, self.k = L, k
        self.cp, self.ap, self.g2 = cp, ap, g2
        self.cSA, self.aSA, self.g3 = cSA, aSA, g3
        self.routIter = routIter

        self.conv1 = nn.Conv1d(
            k, self.cp * self.ap,
            kernel_size=g2, 
            stride=1, 
            padding = 'same') 

        pad = int(g3/2 - 0.5)
        
        self.conv2 = nn.Conv2d(
            1, self.cSA * self.aSA, 
            kernel_size=(self.g3, self.ap), 
            stride=(1, self.ap), 
            padding = (pad, 0))
        

    def forward(self, x):                                                  #[k, L] -
        # primary caps
        x = self.conv1(x)                                                  #[cp*ap, L] -
        x = x.permute(0,2,1).view(x.size(0), self.L, self.cp, self.ap)     #[L, cp*ap] -> [L, cp, ap] -
        x = squash(x)
        x = x.view(x.size(0), self.L, self.cp * self.ap).unsqueeze(1)      #[1, L, cp*ap] -
        
        # time caps
        x = self.conv2(x)                                                              #[cSA*aSA, L, cp] -
        x = x.permute(0,2,3,1).view(x.size(0), self.L, self.cp, self.cSA, self.aSA)    #[L, cp, cSA, aSA] -
        x = routing(x, self.routIter)                                                       #[L, cSA, aSA] -
        
        # flatten
        x = x.view(x.size(0), self.L * self.cSA, self.aSA)                             #[L*cSA, aSA] -
        return x
        
    

class Cell_B(nn.Module):
    def __init__(self, L, k, n, cb, ab, g2, cSB, aSB, g3, routIter):
        super().__init__()
        self.L, self.k, self.n = L, k, n
        self.cb, self.ab, self.g2 = cb, ab, g2
        self.cSB, self.aSB, self.g3 = cSB, aSB, g3
        self.routIter = routIter
        self.Ln = L // n

        self.convLayer = nn.Conv1d(self.k, self.cb, kernel_size=1)
        
        self.conv1 = nn.Conv1d(
            self.cb, self.cb * self.ab, 
            kernel_size=g2, 
            stride=1, 
            padding = 'same')

        pad = int(g3/2 - 0.5)

        self.conv2 = nn.Conv2d(
            1, self.cSB * self.aSB, 
            kernel_size=(self.g3, self.cb * self.ab), 
            stride=(1, cb * ab), 
            padding = (pad, 0))
        

    def forward(self, x):        #[k, L] -
        x = self.convLayer(x)    #[cb, L] -

        # primary caps
        x = self.conv1(x)                                                           #[cb*ab, L] -
        x = x.permute(0,2,1).view(x.size(0), self.Ln, self.n, self.cb * self.ab)    #[l/n, n, cb*ab] -
        x = squash(x)  #squash along the full set of feature maps
        x = x.contiguous().view(x.size(0), self.Ln, self.n * self.cb * self.ab).unsqueeze(1)     #[1, L/n, n*cb*ab] -
        
        # time caps
        x = self.conv2(x)                                                               #[cSB*aSB, L/n, n] -
        x = x.permute(0,2,3,1).view(x.size(0), self.Ln, self.n, self.cSB, self.aSB)     #[L/n, n, cSB, aSB] ? -
        x = routing(x, self.routIter)                                                        #[L/n, cSB, aSB] -
        
        # flatten
        x = x.view(x.size(0), self.Ln * self.cSB, self.aSB)                             #[l/n*cSb, aSB] -
        return x



class ClassificationCapsules(nn.Module):
    def __init__(self, prevNum, prevDim, capsNum, capsDim, routIter):
        super().__init__()
        self.prevNum, self.prevDim = prevNum, prevDim
        self.capsNum, self.capsDim = capsNum, capsDim
        self.routIter = routIter

        self.W = nn.Parameter(0.01 * torch.randn(1, prevNum, capsNum, prevDim, capsDim))

    def forward(self, x):
        x = squash(x)
        x = x.unsqueeze(-2).unsqueeze(-2)
        x = torch.matmul(x, self.W).squeeze(-2)
        x = routing(x, self.routIter)
        return x


class TimeCaps(pl.LightningModule):
    def __init__(self, L, k, n, g1, g2, g3, cp, ap, cSA, aSA, cb, ab, cSB, aSB, routingIterations, n_classes):
        super().__init__()
        self.L, self.k, self.n = L, k, n
        self.g1, self.g2, self.g3 = g1, g2, g3
        self.cp, self.ap, self.cSA, self.aSA = cp, ap, cSA, aSA
        self.cb, self.ab, self.cSB, self.aSB = cb, ab, cSB, aSB
        self.routIter = routingIterations
        self.n_classes = n_classes

        self.test_targets = []
        self.test_preds = []

        assert L % n == 0, '[L] needs to be divisible by [n]'
        assert g3 % 2 != 0, '[g3] needs to be odd'
        assert aSA == aSB, '[aSA] should be equal to [aSB]'

        self.conv1 = nn.Conv1d(
            1, k, 
            kernel_size=g1, 
            stride=1, 
            padding = 'same'
        )
        
        self.cell_A = Cell_A(
            L=self.L, k=self.k, cp=self.cp, ap=self.ap, g2=self.g2, cSA=self.cSA, aSA=self.aSA, g3=self.g3, routIter=self.routIter
        )
        self.cell_B = Cell_B(
            L = self.L, k=self.k, n=self.n, cb=self.cb, ab=self.ab, g2=self.g2, cSB=self.cSB, aSB=self.aSB, g3=self.g3, 
            routIter=self.routIter
        )

        self.classCaps = ClassificationCapsules(
            prevNum=(L*cSA + L//n*cSB), prevDim=aSA, capsNum=n_classes, capsDim=16, routIter=self.routIter)

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
 

    def forward(self, X):
        X = X.unsqueeze(1)
        X = self.conv1(X)   #[k,L] -
        X_A = self.cell_A(X)
        X_B = self.cell_B(X)
        X = self.concat(X_A, X_B)
        X = self.classCaps(X)
        return X


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-6) 

    
    def concat(self, X_A, X_B):  #[b, nCaps, dimCaps]
        return torch.cat((X_A * self.alpha, X_B * self.beta), 1)
    
    # margin loss
    def lossF(self, out, y, m_plus=0.8, m_minus=0.2, λ=0.5):
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
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        preds = torch.norm(out, dim=-1).argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True)

        return loss   
        
    def on_train_epoch_end(self):
        if (self.current_epoch % 10 == 0):
            avg_loss = self.trainer.callback_metrics["train_loss"].item()
            train_acc = self.trainer.callback_metrics["train_acc"].item()
            print(f"Epoch: {self.current_epoch}, Loss: {avg_loss:.4f}, Accuracy: {train_acc*100:.2f}%")


# ------------------- Test ---------------------------------------------
    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)  # [batch, capsNum, capsDim]
        
        # loss
        loss = self.lossF(out, y)
        self.log("test_loss", loss)  
        
        # acc
        preds = torch.norm(out, dim=-1).argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("test_acc", acc)

        self.test_preds.extend(preds.cpu().numpy())  # report rabi np objekt na cpu
        self.test_targets.extend(y.cpu().numpy())
        

    def on_test_epoch_end(self):
        test_loss = self.trainer.callback_metrics["test_loss"].item()
        test_acc = self.trainer.callback_metrics["test_acc"].item()
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_acc*100:.2f}%")

        report = classification_report(self.test_targets, self.test_preds, digits=4)
        print("\n=== Classification Report ===")
        print(report)


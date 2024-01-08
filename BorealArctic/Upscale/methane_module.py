import torch
from typing import *
from torch.nn.utils.rnn import PackedSequence
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import numpy as np

def ML_train(model,opt,opt_chamber,loss,causal_stren_raw,train_loader,
             x_chamber,y_chamber,chamber_train_idxs,
             x_means,x_std,y_means,y_std,chamber_tune_epochs,causal_punish_para,mse_train,device):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device=device)
        batch_y = batch_y.to(device=device)
        opt.zero_grad()
        y_pred, alphas, betas = model(batch_x)
        y_pred = y_pred.squeeze(1)
        l = loss(y_pred, batch_y)
        causal_stren = np.repeat(causal_stren_raw, batch_x.size(0), axis=0)
        causal_stren = causal_stren / np.sum(causal_stren, axis=1, keepdims=True)
        causal_stren = torch.Tensor(causal_stren)
        weights_temp = Variable(causal_stren, requires_grad=False).to(device=device)
        betas = betas.squeeze(2)
        causal_loss = F.kl_div(betas.log(), weights_temp, None, None, 'sum')
        l = l + causal_punish_para * causal_loss
        l.backward()
        mse_train += l.item() * batch_x.shape[0]
        opt.step()
    #############################################
    # chamber data
    for chamber_epoch_idx in range(chamber_tune_epochs):
        for chamber_idx in chamber_train_idxs:
            x_chamber_data = x_chamber[chamber_idx]
            x_nan_num = np.sum(np.isnan(x_chamber_data.reshape(-1)))
            if x_nan_num == 0:
                y_chamber_data = y_chamber[chamber_idx]
                x_chamber_data = (x_chamber_data - x_means) / (x_std + pow(10, -6))
                y_chamber_data = (y_chamber_data - y_means) / (y_std + pow(10, -6))
                x_chamber_data = torch.Tensor(x_chamber_data)
                y_chamber_data = torch.Tensor([y_chamber_data])
                opt_chamber.zero_grad()
                y_pred, alphas, betas = model(x_chamber_data)
                y_pred = y_pred.squeeze(1)
                y_mean = torch.mean(y_pred, dim=0, keepdim=True)
                l = loss(y_mean, y_chamber_data)
                l.backward()
                opt_chamber.step()
                print('loss', l.item())
def ML_validate(model,loss,val_loader,min_val_loss,patience,para_path,
             x_chamber,y_chamber,chamber_validate_idxs,
             x_means,x_std,y_means,y_std,device):
    with torch.no_grad():
        model.eval()
        mse_val = 0
        preds = []
        true = []
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device=device)
            batch_y = batch_y.to(device=device)
            output, alphas, betas = model(batch_x)
            output = output.squeeze(1)
            preds.append(output.cpu().numpy())
            true.append(batch_y.cpu().numpy())
            mse_val += loss(output, batch_y).item() * batch_x.shape[0]
        for chamber_idx in chamber_validate_idxs:
            x_chamber_data = x_chamber[chamber_idx]
            x_nan_num = np.sum(np.isnan(x_chamber_data.reshape(-1)))
            if x_nan_num == 0:
                y_chamber_data = y_chamber[chamber_idx]
                x_chamber_data = (x_chamber_data - x_means) / (x_std + pow(10, -6))
                y_chamber_data = (y_chamber_data - y_means) / (y_std + pow(10, -6))
                x_chamber_data = torch.Tensor(x_chamber_data)
                y_chamber_data = torch.Tensor([y_chamber_data])
                y_pred, alphas, betas = model(x_chamber_data)
                y_pred = y_pred.squeeze(1)
                y_mean = torch.mean(y_pred, dim=0, keepdim=True)
                l = loss(y_mean, y_chamber_data)
                mse_val += l.item()
                preds.append(y_mean.cpu().numpy())
                true.append(y_chamber_data.cpu().numpy())

    if min_val_loss > mse_val ** 0.5:
        min_val_loss = mse_val ** 0.5
        print("Saving...")
        torch.save(model.state_dict(), para_path)  # save the parameters
        counter = 0
    else:
        counter += 1
    return counter
def ML_test(model,loss,test_loader,preds,true,alphas,betas,mse_val,
            chamber_input,chamber_output,chamber_ID,chamber_test_idxs,
             x_means,x_std,y_means,y_std,type,device):

    with torch.no_grad():
        model.eval()
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device=device)
            batch_y = batch_y.to(device=device)
            output, a, b = model(batch_x)
            output = output.squeeze(1)
            preds.append(output.cpu().numpy())
            true.append(batch_y.cpu().numpy())
            alphas.append(a.cpu().numpy())
            betas.append(b.cpu().numpy())
            mse_val += loss(output, batch_y).item() * batch_x.shape[0]
    ############################################################
    # test chamber dataset
    with torch.no_grad():
        model.eval()
        x_chamber = chamber_input[type]
        y_chamber = chamber_output[type]
        IDs = chamber_ID[type]
        for chamber_idx in chamber_test_idxs:
            x_chamber_data = x_chamber[chamber_idx]
            ID = IDs[chamber_idx]
            x_nan_num = np.sum(np.isnan(x_chamber_data.reshape(-1)))
            if x_nan_num == 0:
                y_chamber_data = y_chamber[chamber_idx]
                x_chamber_data = (x_chamber_data - x_means) / (x_std + pow(10, -6))
                y_chamber_data = (y_chamber_data - y_means) / (y_std + pow(10, -6))
                x_chamber_data = torch.Tensor(x_chamber_data)
                y_chamber_data = torch.Tensor([y_chamber_data])
                y_pred, a, b = model(x_chamber_data)
                y_pred = y_pred.squeeze(1)
                y_mean = torch.mean(y_pred, dim=0, keepdim=True)
                y_mean = y_mean.detach().cpu().numpy()
                y_chamber_data = y_chamber_data.cpu().numpy()
                preds.append(y_mean)
                true.append(y_chamber_data)
                alphas.append(a.cpu().numpy())
                betas.append(b.cpu().numpy())
    return preds,true,alphas,betas


class VariationalDropout(nn.Module):
    def __init__(self, dropout: float, batch_first: Optional[bool] = False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x

class Causal_ML(torch.nn.Module):
    __constants__ = ["n_units", "input_dim"]
    def __init__(self, input_dim, output_dim, n_units, init_std=0.02,dropout=0,device='cpu',seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        torch.manual_seed(seed)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        torch.manual_seed(seed)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        torch.manual_seed(seed)
        self.W_i = nn.Linear(input_dim*(n_units+1), input_dim*n_units)
        torch.manual_seed(seed)
        self.W_f = nn.Linear(input_dim*(n_units+1), input_dim*n_units)
        torch.manual_seed(seed)
        self.W_o = nn.Linear(input_dim*(n_units+1), input_dim*n_units)
        torch.manual_seed(seed)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1)*init_std)
        torch.manual_seed(seed)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1)*init_std)
        torch.manual_seed(seed)
        self.F_beta = nn.Linear(2*n_units, 1)
        torch.manual_seed(seed)
        self.Phi = nn.Linear(2*n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim
        self.device=device
        self.dropout=VariationalDropout(dropout,batch_first=True)
    def forward(self, x):
        if self.training:
            x = self.dropout(x)
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(device=self.device)#.cuda()
        c_t = torch.zeros(x.shape[0], self.input_dim*self.n_units).to(device=self.device)#.cuda()
        outputs=[]
        for t in range(x.shape[1]):
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_j) + self.b_j)
            inp =  torch.cat([x[:, t, :], h_tilda_t.view(h_tilda_t.shape[0], -1)], dim=1)
            i_t = torch.sigmoid(self.W_i(inp))
            f_t = torch.sigmoid(self.W_f(inp))
            o_t = torch.sigmoid(self.W_o(inp))
            c_t = c_t*f_t + i_t*j_tilda_t.reshape(j_tilda_t.shape[0], -1)
            h_tilda_t = (o_t*torch.tanh(c_t)).view(h_tilda_t.shape[0], self.input_dim, self.n_units)
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) +self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas/torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas*outputs, dim=1)
        hg = torch.cat([g_n, h_tilda_t], dim=2)
        hg=self.dropout(hg)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        betas = betas/torch.sum(betas, dim=1, keepdim=True)
        mean = torch.sum(betas*mu, dim=1)
        return mean, alphas, betas

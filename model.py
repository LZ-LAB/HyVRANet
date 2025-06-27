# -- coding: utf-8 --

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class BaseClass(torch.nn.Module):
    def __init__(self):
        super(BaseClass, self).__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_mrr = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)
        self.best_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_hit1 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)



class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        return

    def forward(self, pred1, tar1):
        pred1 = F.softmax(pred1, dim=1)
        loss = -torch.log(pred1[tar1 == 1]).sum()
        return loss



class HyVRANet(BaseClass):

    def __init__(self, n_ent, n_rel, input_drop, hidden_drop, feature_drop, VarRAC_Size, PosRAC_Size,
                 emb_dim, max_arity, device):
        super(HyVRANet, self).__init__()
        self.loss = MyLoss()
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.device = device
        self.emb_dim = emb_dim
        
        self.max_arity = max_arity
        
        self.input_drop = nn.Dropout(input_drop) # input_drop
        self.hidden_drop = nn.Dropout(hidden_drop) # hidden_drop
        self.feature_drop = nn.Dropout(feature_drop) # feature_drop
 
        self.ent_embeddings = nn.Parameter(torch.Tensor(self.n_ent, self.emb_dim))
        self.rel_embeddings = nn.Parameter(torch.Tensor(self.n_rel, self.emb_dim))
        self.pos_embeddings = nn.Embedding(self.max_arity, self.emb_dim)

        # 初始化GAN组件
        self.generator = Generator(emb_dim=self.emb_dim, hidden_dim=self.emb_dim*2)
        self.discriminator = Discriminator(emb_dim=self.emb_dim*2, hidden_dim=self.emb_dim)
        self.gen_optimizer = None
        self.dis_optimizer = None

        self.PosRAC_Size = PosRAC_Size
        self.VarRAC_Size = VarRAC_Size
        
        self.Rel_PosW = nn.Linear(in_features=self.emb_dim, out_features=self.PosRAC_Size)
        self.Rel_PosInvW = nn.Linear(in_features=self.emb_dim * self.PosRAC_Size, out_features=self.emb_dim)

        self.Rel_W2 = nn.Linear(in_features=self.emb_dim, out_features=1*self.VarRAC_Size)
        self.Rel_W3 = nn.Linear(in_features=self.emb_dim, out_features=2*self.VarRAC_Size)
        self.Rel_W4 = nn.Linear(in_features=self.emb_dim, out_features=3*self.VarRAC_Size)
        self.Rel_W5 = nn.Linear(in_features=self.emb_dim, out_features=4*self.VarRAC_Size)
        self.Rel_W6 = nn.Linear(in_features=self.emb_dim, out_features=5*self.VarRAC_Size)
        self.Rel_W7 = nn.Linear(in_features=self.emb_dim, out_features=6*self.VarRAC_Size)
        self.Rel_W8 = nn.Linear(in_features=self.emb_dim, out_features=7*self.VarRAC_Size)
        self.Rel_W9 = nn.Linear(in_features=self.emb_dim, out_features=8*self.VarRAC_Size)

        self.pool = torch.nn.MaxPool2d((2, 1))
        self.conv_size = self.emb_dim * self.VarRAC_Size // 2

        self.fc_layer = nn.Linear(in_features=self.conv_size, out_features=self.emb_dim)

        self.bn = nn.BatchNorm2d(num_features=1)
        self.Posbn = nn.BatchNorm2d(num_features=1)        
        # self.bn1 = nn.BatchNorm3d(num_features=1)
        # self.bn2 = nn.BatchNorm3d(num_features=4)
        # self.bn3 = nn.BatchNorm2d(num_features=32)
        # self.bn4 = nn.BatchNorm1d(num_features=self.conv_size)
        
        
        
        # self.w_1 = nn.Linear(in_features=self.d_model, out_features=self.hidden_size)
        # self.w_2 = nn.Linear(in_features=self.hidden_size, out_features=self.d_model)
        self.d_model = emb_dim
        self.hidden_size = self.d_model * 2
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        self.w_1 = nn.Conv1d(self.d_model, self.hidden_size, 1)
        self.w_2 = nn.Conv1d(self.hidden_size, self.d_model, 1)

        

        self.register_parameter('b', nn.Parameter(torch.zeros(n_ent)))

        nn.init.xavier_uniform_(self.ent_embeddings.data)
        nn.init.xavier_uniform_(self.rel_embeddings.data)
        nn.init.xavier_uniform_(self.pos_embeddings.weight.data)

        
        nn.init.xavier_uniform_(self.Rel_PosW.weight.data)
        nn.init.xavier_uniform_(self.Rel_PosInvW.weight.data)        
        nn.init.xavier_uniform_(self.Rel_W2.weight.data)
        nn.init.xavier_uniform_(self.Rel_W3.weight.data)
        nn.init.xavier_uniform_(self.Rel_W4.weight.data)
        nn.init.xavier_uniform_(self.Rel_W5.weight.data)
        nn.init.xavier_uniform_(self.Rel_W6.weight.data)
        nn.init.xavier_uniform_(self.Rel_W7.weight.data)
        nn.init.xavier_uniform_(self.Rel_W8.weight.data)
        nn.init.xavier_uniform_(self.Rel_W9.weight.data)
        
        
        # nn.init.xavier_uniform_(self.w_1.weight.data)
        # nn.init.xavier_uniform_(self.w_2.weight.data)
        
        nn.init.kaiming_uniform_(self.w_1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.w_2.weight, mode='fan_in', nonlinearity='relu')



    def AdapRA_Position(self, rel_embedding, ent_embedding, position):
        ''' Adaptive Relation-Aware Position Embedding Module '''
        ent_embedding = ent_embedding.view(ent_embedding.shape[0], 1, 1, -1)
        
        RA_PosConv1 = self.Rel_PosW(rel_embedding).view(ent_embedding.shape[0]*self.PosRAC_Size, 1, 1, 1)
        RA_PosConv2 = self.Rel_PosW(rel_embedding).view(ent_embedding.shape[0]*self.PosRAC_Size, 1, 1, 1)
        RA_PosConv3 = self.Rel_PosW(rel_embedding).view(ent_embedding.shape[0]*self.PosRAC_Size, 1, 1, 1)
        RA_PosConv4 = self.Rel_PosW(rel_embedding).view(ent_embedding.shape[0]*self.PosRAC_Size, 1, 1, 1)
        RA_PosConv5 = self.Rel_PosW(rel_embedding).view(ent_embedding.shape[0]*self.PosRAC_Size, 1, 1, 1)
        RA_PosConv6 = self.Rel_PosW(rel_embedding).view(ent_embedding.shape[0]*self.PosRAC_Size, 1, 1, 1)
        RA_PosConv7 = self.Rel_PosW(rel_embedding).view(ent_embedding.shape[0]*self.PosRAC_Size, 1, 1, 1)
        RA_PosConv8 = self.Rel_PosW(rel_embedding).view(ent_embedding.shape[0]*self.PosRAC_Size, 1, 1, 1)
        RA_PosConv9 = self.Rel_PosW(rel_embedding).view(ent_embedding.shape[0]*self.PosRAC_Size, 1, 1, 1)
        
        x = self.Posbn(ent_embedding)
        x =  x.permute(1, 0, 2, 3)

        if position == 1:
            x = F.conv2d(x, RA_PosConv1, groups=ent_embedding.size(0))
        if position == 2:
            x = F.conv2d(x, RA_PosConv2, groups=ent_embedding.size(0))
        if position == 3:
            x = F.conv2d(x, RA_PosConv3, groups=ent_embedding.size(0))
        if position == 4:
            x = F.conv2d(x, RA_PosConv4, groups=ent_embedding.size(0))
        if position == 5:
            x = F.conv2d(x, RA_PosConv5, groups=ent_embedding.size(0))
        if position == 6:
            x = F.conv2d(x, RA_PosConv6, groups=ent_embedding.size(0))
        if position == 7:
            x = F.conv2d(x, RA_PosConv7, groups=ent_embedding.size(0))
        if position == 8:
            x = F.conv2d(x, RA_PosConv8, groups=ent_embedding.size(0))
        if position == 9:
            x = F.conv2d(x, RA_PosConv9, groups=ent_embedding.size(0))
        
        x = x.contiguous().view(-1, self.PosRAC_Size * self.emb_dim)
        x = self.Rel_PosInvW(x)
        
        x = x.view(-1, 1, self.emb_dim)

        return x




    def AdapRAConv(self, ent_embedding, rel_embedding, arity):
        ''' Adaptive Relation-Aware Convolution Module '''
        
        ent_embedding = ent_embedding.view(ent_embedding.shape[0], 1, -1,  arity-1)
        
        ## self.VarRAC_Size * (arity - 1)
        rad_kernel2 = self.Rel_W2(rel_embedding).view(ent_embedding.size(0)*self.VarRAC_Size, 1, 1, 1)
        rad_kernel3 = self.Rel_W3(rel_embedding).view(ent_embedding.size(0)*self.VarRAC_Size, 1, 1, 2)
        rad_kernel4 = self.Rel_W4(rel_embedding).view(ent_embedding.size(0)*self.VarRAC_Size, 1, 1, 3)
        rad_kernel5 = self.Rel_W5(rel_embedding).view(ent_embedding.size(0)*self.VarRAC_Size, 1, 1, 4)
        rad_kernel6 = self.Rel_W6(rel_embedding).view(ent_embedding.size(0)*self.VarRAC_Size, 1, 1, 5)
        rad_kernel7 = self.Rel_W7(rel_embedding).view(ent_embedding.size(0)*self.VarRAC_Size, 1, 1, 6)
        rad_kernel8 = self.Rel_W8(rel_embedding).view(ent_embedding.size(0)*self.VarRAC_Size, 1, 1, 7)
        rad_kernel9 = self.Rel_W9(rel_embedding).view(ent_embedding.size(0)*self.VarRAC_Size, 1, 1, 8)

        x = ent_embedding
        x = self.bn(x)
        x = x.permute(1, 0, 2, 3)

        if arity == 2:
            x = F.conv2d(x, rad_kernel2, groups=ent_embedding.size(0))
        if arity == 3:
            x = F.conv2d(x, rad_kernel3, groups=ent_embedding.size(0))
        if arity == 4:
            x = F.conv2d(x, rad_kernel4, groups=ent_embedding.size(0))
        if arity == 5:
            x = F.conv2d(x, rad_kernel5, groups=ent_embedding.size(0))
        if arity == 6:
            x = F.conv2d(x, rad_kernel6, groups=ent_embedding.size(0))
        if arity == 7:
            x = F.conv2d(x, rad_kernel7, groups=ent_embedding.size(0))
        if arity == 8:
            x = F.conv2d(x, rad_kernel8, groups=ent_embedding.size(0))          
        if arity == 9:
            x = F.conv2d(x, rad_kernel9, groups=ent_embedding.size(0))
        
        x = self.pool(x)
        x = x.contiguous().view(-1, self.conv_size)
        x = self.feature_drop(x)

        return x


    def AdapRAConv_Process(self, concat_input):
        ''' The Feature Process of Adaptive Relation-Aware Convolution Module '''        
        r = concat_input[:, 0, :]
        
        if concat_input.shape[1] == 2:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim)

            e1 = self.AdapRA_Position(r, e1, position = 1) + e1
            
            ent_features = e1
            x = self.AdapRAConv(ent_features, r, arity = 2)
                    
        if concat_input.shape[1] == 3:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim)
            
            e1 = self.AdapRA_Position(r, e1, position = 1) + e1
            e2 = self.AdapRA_Position(r, e2, position = 2) + e2

            ent_features = torch.cat((e1, e2), dim=1)
            x = self.AdapRAConv(ent_features, r, arity = 3)

        if concat_input.shape[1] == 4:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim)
            
            e1 = self.AdapRA_Position(r, e1, position = 1) + e1
            e2 = self.AdapRA_Position(r, e2, position = 2) + e2
            e3 = self.AdapRA_Position(r, e3, position = 3) + e3
            
            ent_features = torch.cat((e1, e2, e3), dim=1)
            x = self.AdapRAConv(ent_features, r, arity = 4)
 
        if concat_input.shape[1] == 5:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim)
            e4 = concat_input[:, 4, :].view(-1, 1, self.emb_dim)
            
            e1 = self.AdapRA_Position(r, e1, position = 1) + e1
            e2 = self.AdapRA_Position(r, e2, position = 2) + e2
            e3 = self.AdapRA_Position(r, e3, position = 3) + e3
            e4 = self.AdapRA_Position(r, e4, position = 4) + e4
            
            ent_features = torch.cat((e1, e2, e3, e4), dim=1)           
            x = self.AdapRAConv(ent_features, r, arity = 5)
            
        if concat_input.shape[1] == 6:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim)
            e4 = concat_input[:, 4, :].view(-1, 1, self.emb_dim)
            e5 = concat_input[:, 5, :].view(-1, 1, self.emb_dim)
            
            e1 = self.AdapRA_Position(r, e1, position = 1) + e1
            e2 = self.AdapRA_Position(r, e2, position = 2) + e2
            e3 = self.AdapRA_Position(r, e3, position = 3) + e3
            e4 = self.AdapRA_Position(r, e4, position = 4) + e4
            e5 = self.AdapRA_Position(r, e5, position = 5) + e5
                        
            ent_features = torch.cat((e1, e2, e3, e4, e5), dim=1)           
            x = self.AdapRAConv(ent_features, r, arity = 6)
            
        if concat_input.shape[1] == 7:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim)
            e4 = concat_input[:, 4, :].view(-1, 1, self.emb_dim)
            e5 = concat_input[:, 5, :].view(-1, 1, self.emb_dim)
            e6 = concat_input[:, 6, :].view(-1, 1, self.emb_dim)
            
            e1 = self.AdapRA_Position(r, e1, position = 1) + e1
            e2 = self.AdapRA_Position(r, e2, position = 2) + e2
            e3 = self.AdapRA_Position(r, e3, position = 3) + e3
            e4 = self.AdapRA_Position(r, e4, position = 4) + e4
            e5 = self.AdapRA_Position(r, e5, position = 5) + e5
            e6 = self.AdapRA_Position(r, e6, position = 6) + e6
            
            ent_features = torch.cat((e1, e2, e3, e4, e5, e6), dim=1)           
            x = self.AdapRAConv(ent_features, r, arity = 7)            

        if concat_input.shape[1] == 8:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim)
            e4 = concat_input[:, 4, :].view(-1, 1, self.emb_dim)
            e5 = concat_input[:, 5, :].view(-1, 1, self.emb_dim)
            e6 = concat_input[:, 6, :].view(-1, 1, self.emb_dim)
            e7 = concat_input[:, 7, :].view(-1, 1, self.emb_dim)
            
            e1 = self.AdapRA_Position(r, e1, position = 1) + e1
            e2 = self.AdapRA_Position(r, e2, position = 2) + e2
            e3 = self.AdapRA_Position(r, e3, position = 3) + e3
            e4 = self.AdapRA_Position(r, e4, position = 4) + e4
            e5 = self.AdapRA_Position(r, e5, position = 5) + e5
            e6 = self.AdapRA_Position(r, e6, position = 6) + e6
            e7 = self.AdapRA_Position(r, e7, position = 7) + e7
                       
            ent_features = torch.cat((e1, e2, e3, e4, e5, e6, e7), dim=1)           
            x = self.AdapRAConv(ent_features, r, arity = 8)
        
        if concat_input.shape[1] == 9:
            e1 = concat_input[:, 1, :].view(-1, 1, self.emb_dim)
            e2 = concat_input[:, 2, :].view(-1, 1, self.emb_dim)
            e3 = concat_input[:, 3, :].view(-1, 1, self.emb_dim)
            e4 = concat_input[:, 4, :].view(-1, 1, self.emb_dim)
            e5 = concat_input[:, 5, :].view(-1, 1, self.emb_dim)
            e6 = concat_input[:, 6, :].view(-1, 1, self.emb_dim)
            e7 = concat_input[:, 7, :].view(-1, 1, self.emb_dim)
            e8 = concat_input[:, 8, :].view(-1, 1, self.emb_dim)
            
            e1 = self.AdapRA_Position(r, e1, position = 1) + e1
            e2 = self.AdapRA_Position(r, e2, position = 2) + e2
            e3 = self.AdapRA_Position(r, e3, position = 3) + e3
            e4 = self.AdapRA_Position(r, e4, position = 4) + e4
            e5 = self.AdapRA_Position(r, e5, position = 5) + e5
            e6 = self.AdapRA_Position(r, e6, position = 6) + e6
            e7 = self.AdapRA_Position(r, e7, position = 7) + e7
            e8 = self.AdapRA_Position(r, e8, position = 8) + e8
            
            ent_features = torch.cat((e1, e2, e3, e4, e5, e6, e7, e8), dim=1)           
            x = self.AdapRAConv(ent_features, r, arity = 9)
        
        x = self.hidden_drop(x)

        return x






    def HyVRANet(self, concat_input):        
        v_out = self.AdapRAConv_Process(concat_input)
        x = self.fc_layer(v_out)
        residual = x
        x = x.transpose(0, 1)
        x = self.w_2(F.relu(self.w_1(x)))
        x = x.transpose(0, 1)
        x = self.layer_norm(x + residual)

        return x





    def forward(self, rel_idx, ent_idx, miss_ent_domain):

        r = self.rel_embeddings[rel_idx].unsqueeze(1)
        ents = self.ent_embeddings[ent_idx]
        
        concat_input = torch.cat((r, ents), dim=1)   
        concat_input = self.input_drop(concat_input) # input_drop    
        
        x = self.HyVRANet(concat_input)

        miss_ent_domain = torch.LongTensor([miss_ent_domain-1]).to(self.device)
        mis_pos = self.pos_embeddings(miss_ent_domain)
        tar_emb = self.ent_embeddings + mis_pos
        
        scores = torch.mm(x, tar_emb.transpose(0, 1))
        scores += self.b.expand_as(scores)

        return scores








    def init_gan_optimizers(self, gen_lr, dis_lr):
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr

        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.gen_lr)
        self.dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.dis_lr)
        






    def generate_gan_negative_samples(self, rel_idx, ent_idx, miss_ent_domain, num_neg=5):
        with torch.no_grad():
            r = self.rel_embeddings[rel_idx].unsqueeze(1)
            ents = self.ent_embeddings[ent_idx]
            concat_input = torch.cat((r, ents), dim=1)
            
            miss_ent_domain_tensor = torch.LongTensor([miss_ent_domain-1]).to(self.device)
            mis_pos = self.pos_embeddings(miss_ent_domain_tensor)
            
            batch_size = rel_idx.size(0)
            neg_samples = []
            
            for _ in range(num_neg):
                random_ents = torch.randint(0, self.n_ent, (batch_size,), device=self.device)
                random_embs = self.ent_embeddings[random_ents]
                
                noise_factor = 0.1
                neg_embs = self.generator(random_embs, noise_factor)
                
                sim_scores = torch.mm(neg_embs, self.ent_embeddings.t())
                
                _, neg_ents = torch.topk(sim_scores, 1, dim=1)
                neg_ents = neg_ents.squeeze(1)
                
                neg_samples.append(neg_ents)
                
            if batch_size == 1:
                return torch.stack([ns.view(1) for ns in neg_samples], dim=1)
            else:
                return torch.stack(neg_samples, dim=1)




    def train_gan(self, rel_idx, ent_idx, miss_ent_domain, pos_ent_idx):
        if self.gen_optimizer is None or self.dis_optimizer is None:
            return 0, 0
            
        batch_size = rel_idx.size(0)
        
        valid_mask = pos_ent_idx < self.n_ent
        if not torch.all(valid_mask):
            valid_indices = torch.where(valid_mask)[0]
            if len(valid_indices) == 0:
                return 0, 0
                
            rel_idx = rel_idx[valid_indices]
            ent_idx = ent_idx[valid_indices]
            pos_ent_idx = pos_ent_idx[valid_indices]
            batch_size = len(valid_indices)
        
        r = self.rel_embeddings[rel_idx].unsqueeze(1)
        ents = self.ent_embeddings[ent_idx]
        concat_input = torch.cat((r, ents), dim=1)
        x = self.HyVRANet(concat_input)
        
        pos_embs = self.ent_embeddings[pos_ent_idx]
        
        random_ents = torch.randint(0, self.n_ent, (batch_size,), device=self.device)
        random_embs = self.ent_embeddings[random_ents]
        gen_neg_embs = self.generator(random_embs)
        
        self.dis_optimizer.zero_grad()
        
        real_pairs = torch.cat([x, pos_embs], dim=1)
        real_labels = torch.ones(batch_size, 1).to(self.device)
        real_preds = self.discriminator(real_pairs)
        d_real_loss = F.binary_cross_entropy(real_preds, real_labels)
        
        fake_pairs = torch.cat([x, gen_neg_embs], dim=1)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        fake_preds = self.discriminator(fake_pairs.detach())
        d_fake_loss = F.binary_cross_entropy(fake_preds, fake_labels)
        

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward(retain_graph=True)
        self.dis_optimizer.step()
        

        self.gen_optimizer.zero_grad()
        

        fake_pairs = torch.cat([x, gen_neg_embs], dim=1)
        fake_preds = self.discriminator(fake_pairs)
        g_loss = F.binary_cross_entropy(fake_preds, real_labels)
        
        g_loss.backward()
        self.gen_optimizer.step()
        
        return g_loss.item(), d_loss.item()
    





class Generator(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super(Generator, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, emb_dim)
        
        self.dropout = nn.Dropout(0.2)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, noise_factor=0.1):
        noise = torch.randn_like(x) * noise_factor
        x = x + noise
        
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class Discriminator(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.emb_dim = emb_dim
        
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim // 2)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x
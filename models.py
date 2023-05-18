# Reference:
# https://github.com/easezyc/WSDM2022-PTUPCDR

import torch
import torch.nn.functional as F


class LookupEmbedding(torch.nn.Module):
  def __init__(self, uid_all, iid_all, emb_dim):
    super().__init__()
    self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
    self.iid_embedding = torch.nn.Embedding(iid_all + 1, emb_dim)

  def forward(self, x):
    uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1))
    iid_emb = self.iid_embedding(x[:, 1].unsqueeze(1))
    emb = torch.cat([uid_emb, iid_emb], dim=1)
    return emb


class DeepMetaNet(torch.nn.Module):
  def __init__(self, emb_dim, meta_dim):
    super().__init__()
    self.event_K = torch.nn.Sequential(torch.nn.Linear(emb_dim, 7), torch.nn.ELU(alpha = 0.8), torch.nn.Dropout(p=0.1),
                                      torch.nn.Linear(7, 4), torch.nn.ELU(alpha = 0.8), torch.nn.Dropout(p=0.1),
                                      torch.nn.Linear(4, 2), torch.nn.ELU(alpha = 0.8), torch.nn.Dropout(p=0.1),
                                      torch.nn.Linear(2, 1, False))
    self.event_softmax = torch.nn.Softmax(dim=1)
    self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim, 25), torch.nn.ELU(alpha = 0.8), torch.nn.Dropout(p=0.2),
                                      torch.nn.Linear(25, 50), torch.nn.ELU(alpha = 0.8), torch.nn.Dropout(p=0.2),
                                      torch.nn.Linear(50, 75), torch.nn.ELU(alpha = 0.8), torch.nn.Dropout(p=0.2),
                                      torch.nn.Linear(75, emb_dim * emb_dim))

  def forward(self, emb_fea, seq_index):
    mask = (seq_index == 0).float()
    event_K = self.event_K(emb_fea)
    t = event_K - torch.unsqueeze(mask, 2) * 1e8
    att = self.event_softmax(t)
    his_fea = torch.sum(att * emb_fea, 1)
    output = self.decoder(his_fea)
    return output.squeeze(1)


class Model(torch.nn.Module):
  def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim_0):
    super().__init__()
    self.num_fields = num_fields
    self.emb_dim = emb_dim
    self.src_model = LookupEmbedding(uid_all, iid_all, emb_dim)
    self.tgt_model = LookupEmbedding(uid_all, iid_all, emb_dim)
    self.aug_model = LookupEmbedding(uid_all, iid_all, emb_dim)
    self.deep_meta_net = DeepMetaNet(emb_dim, meta_dim_0)
    self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)

  def forward(self, x, stage):
    if stage == 'train_src':
      emb = self.src_model.forward(x)
      x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
      return x
    elif stage in ['train_tgt', 'test_tgt']:
      emb = self.tgt_model.forward(x)
      x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
      return x
    elif stage in ['train_aug', 'test_aug']:
      emb = self.aug_model.forward(x)
      x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
      return x
    elif stage in ['train_meta', 'test_meta']:
      iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))
      uid_emb_src = self.src_model.uid_embedding(x[:, 0].unsqueeze(1))
      ufea = self.src_model.iid_embedding(x[:, 2:])
      mapping = self.deep_meta_net.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
      uid_emb = torch.bmm(uid_emb_src, mapping)
      emb = torch.cat([uid_emb, iid_emb], 1)
      output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
      return output
    elif stage == 'train_map':
      src_emb = self.src_model.uid_embedding(x.unsqueeze(1)).squeeze()
      src_emb = self.mapping.forward(src_emb)
      tgt_emb = self.tgt_model.uid_embedding(x.unsqueeze(1)).squeeze()
      return src_emb, tgt_emb
    elif stage == 'test_map':
      uid_emb = self.mapping.forward(self.src_model.uid_embedding(x[:, 0].unsqueeze(1)).squeeze())
      emb = self.tgt_model.forward(x)
      emb[:, 0, :] = uid_emb
      x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
      return x

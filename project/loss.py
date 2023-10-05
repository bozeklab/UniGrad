import torch
from torch.nn import functional as F


class Contrast(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        device = torch.device(f"cuda:{cfg.local_rank}")
        self.batch_size = cfg.batch_size
        self.register_buffer("temp", torch.tensor(cfg.temperature).to(torch.device(f"cuda:{cfg.local_rank}")))
        self.register_buffer("neg_mask", (~torch.eye(cfg.batch_size * 2,
                                                     cfg.batch_size * 2, dtype=bool).to(device)).float())

    def forward(self, x_i, x_j):
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim, self.batch_size)
        sim_ji = torch.diag(sim, -self.batch_size)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)
        denom = self.neg_mask * torch.exp(sim / self.temp)
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size)
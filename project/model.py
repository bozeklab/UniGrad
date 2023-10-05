import copy
import math
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import LayerNorm

from util.pos_embed import interpolate_pos_embed

from project.unetr_vits import unetr_vit_base_patch16, cell_vit_base_patch16


def prepare_unetr_model(chkpt_dir_vit, **kwargs):
    # build ViT encoder
    num_nuclei_classes = kwargs.pop('num_nuclei_classes')
    num_tissue_classes = kwargs.pop('num_tissue_classes')
    embed_dim = kwargs.pop('embed_dim')
    extract_layers = kwargs.pop('extract_layers')
    drop_rate = kwargs['drop_path_rate']

    vit_encoder = unetr_vit_base_patch16(num_classes=num_tissue_classes, **kwargs)

    # load ViT model
    checkpoint = torch.load(chkpt_dir_vit, map_location='cpu')

    checkpoint_model = checkpoint['model']
    interpolate_pos_embed(vit_encoder, checkpoint_model)

    msg = vit_encoder.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    # assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

    model = cell_vit_base_patch16(num_nuclei_classes=num_nuclei_classes,
                                  embed_dim=embed_dim,
                                  extract_layers=extract_layers,
                                  drop_rate=drop_rate,
                                  encoder=vit_encoder)
    model.freeze_encoder()
    model.cuda()
    return model


def build_model(cfg):
    if cfg.siamese == True:
        model = SiameseNet(cfg)
    else:
        model = OneHeadNet(cfg)
    if torch.distributed.is_available():
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda()
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[torch.cuda.current_device()],
                                                          find_unused_parameters=True)
    else:
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    return model


class OneHeadNet(nn.Module):

    def __init__(self, cfg):
        super(OneHeadNet, self).__init__()
        self.cfg = cfg

        self.encoder = prepare_unetr_model(chkpt_dir_vit=cfg.encoder_path,
                                           init_values=None,
                                           drop_path_rate=cfg.drop_path_rate,
                                           num_nuclei_classes=6,
                                           num_tissue_classes=19,
                                           embed_dim=cfg.embed_dim,
                                           extract_layers=[3, 6, 9, 12])

        # build online branch
        # self.encoder = nn.Sequential(*list(net.children())[:-1] + [nn.Flatten(1)])
        #self.projector = _projection_mlp(self.cfg.projector_input_dim,
        #                                self.cfg.projector_hidden_dim,
        #                                self.cfg.projector_output_dim)

    def forward(self, x1, x2, boxes1, boxes2, mask):
        y1 = self.encoder(x1, boxes1, mask)
        y2 = self.encoder(x2, boxes2, mask)

        z1 = self.projector(y1)
        z2 = self.projector(y2)

        return z1, z2


class SiameseNet(nn.Module):
    """
    Build a Siamese model.
    Some codes are borrowed from MoCo & SimSiam.
    """

    def __init__(self, cfg):
        super(SiameseNet, self).__init__()
        self.cfg = cfg

        zero_init_residual = getattr(self.cfg, 'zero_init_residual', True)
        self.encoder = prepare_unetr_model(chkpt_dir_vit=cfg.encoder_path,
                                           init_values=None,
                                           drop_path_rate=cfg.drop_path_rate,
                                           num_nuclei_classes=6,
                                           num_tissue_classes=19,
                                           embed_dim=cfg.embed_dim,
                                           extract_layers=[3, 6, 9, 12])

        # build online branch
        # self.encoder = nn.Sequential(*list(net.children())[:-1] + [nn.Flatten(1)])
        #self.projector = _projection_mlp(self.cfg.projector_input_dim,
        #                                 self.cfg.projector_hidden_dim,
        #                                 self.cfg.projector_output_dim)

        # build target branch
        self.momentum_encoder = copy.deepcopy(self.encoder)
        #self.momentum_projector = copy.deepcopy(self.projector)
        self.teacher_norm = LayerNorm(self.cfg.projector_input_dim, elementwise_affine=False)
        #self.student_norm = LayerNorm(self.cfg.projector_output_dim)
        #for p in self.student_norm.parameters():
        #    p.requires_grad = False
        for p in self.teacher_norm.parameters():
            p.requires_grad = False
        self.student_norm = nn.Identity()

    @torch.no_grad()
    def _momentum_update_key_encoder(self, mm):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        #for param_q, param_k in zip(self.projector.parameters(), self.momentum_projector.parameters()):
        #    param_k.data = param_k.data * mm + param_q.data * (1. - mm)

    def forward(self, x1, x2, boxes1, boxes2, mask, mm):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https:/ /arxiv.org/abs/2011.10566 for detailed notations
        """
        # online branch

        pred = self.encoder(x1, boxes1, mask)
        #pred = self.projector(y1)
        pred = self.student_norm(pred)

        # target branch
        with torch.no_grad():
            self._momentum_update_key_encoder(mm)
            target = self.momentum_encoder(x2, boxes2, mask)
            #target = self.momentum_projector(y2m)
            target = self.teacher_norm(target)
        return pred, target


def _projection_mlp(in_dims: int,
                    h_dims: int,
                    out_dims: int,
                    bias: bool = False) -> nn.Sequential:
    """Projection MLP. The original paper's implementation has 3 layers, with
    BN applied to its hidden fc layers but no ReLU on the output fc layer.
    The CIFAR-10 study used a MLP with only two layers.

    Args:
        in_dims:
            Input dimension of the first linear layer.
        h_dims:
            Hidden dimension of all the fully connected layers.
        out_dims:
            Output Dimension of the final linear layer.

    Returns:
        nn.Sequential:
            The projection head.
    """
    l1 = nn.Sequential(nn.Linear(in_dims, h_dims, bias=bias),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l2 = nn.Sequential(nn.Linear(h_dims, h_dims, bias=bias),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l3 = nn.Sequential(nn.Linear(h_dims, out_dims))

    projection = nn.Sequential(l1, l2, l3)

    return projection
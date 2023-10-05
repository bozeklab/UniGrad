seed: 42 # random set
launcher: pytorch # pytorch or slurm
port: 28500 # distributed port

# dataset
train_datadir: '/data/pwojcik/pannuke_contrastive/fold_1_3_256/positive/'
test_datadir: '/data/pwojcik/pannuke_contrastive/fold_2_256/positive/'
encoder_path: '/data/pwojcik/SimMIM/output_dir_sim1600_skin/sim_base_1600ep_pretrain.pth'
n_workers: 5
input_size: 256
num_boxes: 150
crop_min: 0.2

# model
arch: 'resnet50'
siamese: True
projector_input_dim: 768
projector_hidden_dim: 1024
projector_output_dim: 1024
drop_rate: 0
attn_drop_rate: 0.1
drop_path_rate: 0.1
embed_dim: 768
num_layers: 3
base_momentum: 0.996
resume_path:

# optimizer
base_lr: 6.25e-5

whole_batch_size: 128
momentum: 0.9
weight_decay: 1.0e-4
epochs: 100
warmup_epochs: 5

# loss
loss: 'loss_unigrad2'
neg_weight: 0.02
lambd:      # balance factor for neg grad
rho:        # for byol & unigrad only
eps:        # for byol & unigrad only
moco_t: 0.2    # for simclr
temperature: 0.5

# others
print_freq: 50
test_freq: 10
save_freq: 10

# knn config
knn_k: 200
knn_t: 0.1
knn_eval: False
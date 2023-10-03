seed: 42 # random set
launcher: pytorch # pytorch or slurm
port: 28500 # distributed port

# dataset
train_datadir: '/home/pwojcik/pwojcik/pannuke/fold_1_3_256/positive/'
test_datadir: '/home/pwojcik/pwojcik/pannuke/fold_2_256/positive/'
encoder_path: '/scratch/pwojcik/encoder-1600.pth'
n_workers: 5
input_size: 256
num_boxes: 150
crop_min: 0.2

# model
arch: 'resnet50'
projector_input_dim: 768
projector_hidden_dim: 2048
projector_output_dim: 2048
drop_rate: 0
attn_drop_rate: 0.1
drop_path_rate: 0.1
embed_dim: 768
num_layers: 3
base_momentum: 0.996
resume_path: 

# optimizer
base_lr: 0.05
whole_batch_size: 256
momentum: 0.9
weight_decay: 1.0e-4
epochs: 100
warmup_epochs: 5

# loss
loss: 'loss_simclr_grad_mm'
lambd:      # balance factor for neg grad
rho:        # for byol & unigrad only
eps:        # for byol & unigrad only
moco_t: 0.2    # for simclr

# others
print_freq: 50
test_freq: 10
save_freq: 10

# knn config
knn_k: 200
knn_t: 0.1
knn_eval: False
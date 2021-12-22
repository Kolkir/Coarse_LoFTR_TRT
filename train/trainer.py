import os

import cv2
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as torch_func

from train.mvsdataset import MVSDataset
from train.saveutils import load_last_checkpoint, save_checkpoint
from utils import get_coarse_match
from loftr import LoFTR, default_cfg
from utils import make_student_config
from webcam import draw_features


def softmax_with_temperature(x, t=1.0):
    ex = torch.exp(x / t)
    ex_sum = torch.sum(ex)
    return ex / ex_sum


def cross_entropy(input, target):
    return -torch.sum(target * torch.log(input))


def tensor_to_image(image):
    frame = image[0, :, :, :].cpu().numpy()
    res_img = (frame * 255.).astype('uint8')
    res_img = np.transpose(res_img, [1, 2, 0])  # OpenCV format
    res_img = cv2.UMat(res_img)
    return res_img.get()


class Trainer(object):
    def __init__(self, settings, weights_path, dataset_path, checkpoint_path):
        self.settings = settings
        self.checkpoint_path = checkpoint_path
        self.learning_rate = self.settings.learning_rate
        self.epochs = self.settings.epochs
        self.summary_writer = None
        if self.settings.write_statistics:
            self.summary_writer = SummaryWriter(log_dir=os.path.join(checkpoint_path, 'runs'))
        self.optimizer = None

        self.global_train_index = 0
        self.last_image1 = None
        self.last_image2 = None
        self.last_teacher_conf_matrix = None
        self.last_student_conf_matrix = None

        print(f'Trainer is initialized with batch size = {self.settings.batch_size}')
        print(f'Gradient accumulation batch size divider = {self.settings.batch_size_divider}')
        print(f'Automatic Mixed Precision = {self.settings.use_amp}')

        self.scaler = torch.cuda.amp.GradScaler(init_scale=self.settings.amp_scale)

        real_batch_size = self.settings.batch_size // self.settings.batch_size_divider

        self.teacher_cfg = default_cfg
        self.teacher_cfg['input_batch_size'] = real_batch_size
        self.teacher_model = LoFTR(config=self.teacher_cfg)
        checkpoint = torch.load(weights_path)
        if checkpoint is not None:
            missed_keys, unexpected_keys = self.teacher_model.load_state_dict(checkpoint['state_dict'], strict=False)
            if len(missed_keys) > 0:
                print('Checkpoint is broken')
                exit(1)
            print('Teachers pre-trained weights were successfully loaded.')
        else:
            print('Failed to load checkpoint')

        self.student_cfg = make_student_config(default_cfg)
        self.student_cfg['input_batch_size'] = real_batch_size
        self.student_model = LoFTR(config=self.student_cfg)

        if self.settings.cuda:
            self.teacher_model = self.teacher_model.cuda()
            self.student_model = self.student_model.cuda()

        if self.settings.write_statistics:
            self.add_model_graph()

        # setup dataset
        batch_size = self.settings.batch_size // self.settings.batch_size_divider
        self.train_dataset = MVSDataset(dataset_path,
                                        (self.student_cfg['input_width'], self.student_cfg['input_height']),
                                        self.student_cfg['resolution'][0],
                                        epoch_size=5000)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=self.settings.data_loader_num_workers)

        # self.create_optimizer()
        self.create_default_optimizer()

    def add_image_summary(self, name, image1, image2,
                          teacher_conf_matrix,
                          teacher_config,
                          student_conf_matrix,
                          student_config):
        assert (teacher_config['input_height'] == student_config['input_height'])
        assert (teacher_config['input_width'] == student_config['input_width'])
        img_size = (teacher_config['input_width'], teacher_config['input_height'])
        image1 = tensor_to_image(image1)
        image2 = tensor_to_image(image2)

        def draw_feature_points(conf_matrix, config, color):
            conf_matrix = conf_matrix.detach().cpu().numpy()
            mkpts0, mkpts1, mconf = get_coarse_match(conf_matrix, config['input_height'], config['input_width'],
                                                     config['resolution'][0])
            # filter only the most confident features
            n_top = 20
            indices = np.argsort(mconf)[::-1]
            indices = indices[:n_top]
            mkpts0 = mkpts0[indices, :]
            mkpts1 = mkpts1[indices, :]

            draw_features(image1, mkpts0, img_size, color)
            draw_features(image2, mkpts1, img_size, color)

        if self.settings.with_teacher:
            draw_feature_points(teacher_conf_matrix[0, :, :].unsqueeze(0), teacher_config, (255, 255, 255))
        draw_feature_points(student_conf_matrix[0, :, :].unsqueeze(0), student_config, (0, 0, 0))

        # combine images
        res_img = np.hstack((image1, image2))
        res_img = res_img[None]

        self.summary_writer.add_image(f'{name} result/train', res_img,
                                      self.global_train_index)

    def train_loop(self):
        train_total_loss = torch.tensor(0., device='cuda' if self.settings.cuda else 'cpu')
        divider = torch.tensor(self.settings.batch_size_divider, device='cuda' if self.settings.cuda else 'cpu')
        real_batch_index = 0
        progress_bar = tqdm(self.train_dataloader)
        # torch.autograd.set_detect_anomaly(True)
        for batch_index, batch in enumerate(progress_bar):
            with torch.set_grad_enabled(True):
                if self.settings.use_amp:
                    with torch.cuda.amp.autocast():
                        losses = self.train_loss_fn(*batch)
                        # normalize loss to account for batch accumulation
                        for loss in losses:
                            loss /= divider
                        loss = torch.stack(losses).sum()

                    # Scales the loss, and calls backward()
                    # to create scaled gradients
                    self.scaler.scale(loss).backward()
                    train_total_loss += loss.detach()
                else:
                    losses = self.train_loss_fn(*batch)
                    # normalize loss to account for batch accumulation
                    for loss in losses:
                        loss /= divider
                    loss = torch.stack(losses).sum()
                    loss.backward()
                    train_total_loss += loss.detach()

                # gradient accumulation
                if ((batch_index + 1) % self.settings.batch_size_divider == 0) or (
                        batch_index + 1 == len(self.train_dataloader)):

                    current_total_loss = train_total_loss / real_batch_index

                    # for logging unscaled grads
                    # if self.settings.use_amp:
                    #     self.scaler.unscale_(self.optimizer)

                    # save statistics
                    if self.settings.write_statistics:
                        self.write_batch_statistics(real_batch_index)

                    if (real_batch_index + 1) % self.settings.statistics_period == 0:
                        cur_lr = [group['lr'] for group in self.optimizer.param_groups]
                        progress_bar.set_postfix(
                            {'Total loss': current_total_loss.item(),
                             'Learning rate': cur_lr})

                    # Optimizer step - apply gradients
                    if self.settings.use_amp:
                        # Unscales gradients and calls or skips optimizer.step()
                        self.scaler.step(self.optimizer)
                        # Updates the scale for next iteration
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    # Clear gradients
                    # This does not zero the memory of each individual parameter,
                    # also the subsequent backward pass uses assignment instead of addition to store gradients,
                    # this reduces the number of memory operations -compared to optimizer.zero_grad()
                    self.optimizer.zero_grad(set_to_none=True)

                    self.global_train_index += 1
                    real_batch_index += 1

        train_loss = train_total_loss.item() / real_batch_index
        return train_loss

    def create_optimizer(self):
        def exclude(n):
            return "bn" in n or "bias" in n or "identity" in n

        def include(n):
            return not exclude(n)

        named_parameters = list(self.student_model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": self.settings.optimizer_weight_decay},
            ],
            lr=self.settings.learning_rate,
            betas=(self.settings.optimizer_beta1, self.settings.optimizer_beta2),
            eps=self.settings.optimizer_eps,
        )

    def create_default_optimizer(self):
        parameters = self.student_model.parameters()

        self.optimizer = torch.optim.AdamW(
            params=parameters,
            lr=self.settings.learning_rate,
        )

    def add_model_graph(self):
        img_size = (
            self.student_cfg['input_batch_size'], 1, self.student_cfg['input_height'], self.student_cfg['input_width'])
        fake_input = torch.ones(img_size, dtype=torch.float32)
        if self.settings.cuda:
            fake_input = fake_input.cuda()
        self.summary_writer.add_graph(self.student_model, [fake_input, fake_input])
        self.summary_writer.flush()

    def train(self, name):
        # continue training starting from the latest epoch checkpoint
        start_epoch = 0
        prev_epoch = load_last_checkpoint(self.checkpoint_path, self.student_model, self.optimizer, self.scaler)
        if prev_epoch >= 0:
            start_epoch = prev_epoch + 1
        epochs_num = start_epoch + self.epochs

        self.global_train_index = 0

        self.teacher_model.eval()
        self.student_model.train()
        for epoch in range(start_epoch, epochs_num):
            print(f"Epoch {epoch}\n-------------------------------")
            train_loss = self.train_loop()
            print(f"Train Loss:{train_loss:7f} \n")
            if self.settings.write_statistics:
                self.summary_writer.add_scalar('Loss/train', train_loss, epoch)

            save_checkpoint(name, epoch, self.student_model, self.optimizer, self.scaler, self.checkpoint_path)
            self.train_dataset.reset_epoch()

    def write_batch_statistics(self, batch_index):
        if (batch_index + 1) % self.settings.statistics_period == 0:
            for name, param in self.student_model.named_parameters():
                if param.grad is not None and 'bn' not in name and 'bias' not in name:
                    if not torch.isnan(param.grad).any():
                        self.summary_writer.add_histogram(
                            tag=f"params/{name}", values=param, global_step=self.global_train_index
                        )
                        self.summary_writer.add_histogram(
                            tag=f"grads/{name}", values=param.grad, global_step=self.global_train_index
                        )

            if self.last_image1 is not None and self.last_image1 is not None:
                self.add_image_summary('Teacher+Student', self.last_image1, self.last_image2,
                                       self.last_teacher_conf_matrix,
                                       self.teacher_cfg,
                                       self.last_student_conf_matrix,
                                       self.student_cfg
                                       )

    def train_loss_fn(self, image1, image2, conf_matrix_gt):
        if self.settings.cuda:
            image1 = image1.cuda()
            image2 = image2.cuda()

        student_conf_matrix = self.student_model.forward(image1, image2)

        if self.settings.with_teacher:
            with torch.no_grad():
                teacher_conf_matrix = self.teacher_model.forward(image1, image2)
            scale = self.student_cfg['resolution'][0] // self.teacher_cfg['resolution'][0]
            i_ids = torch.arange(start=0, end=student_conf_matrix.shape[1], device=student_conf_matrix.device) * scale
            j_ids = torch.arange(start=0, end=student_conf_matrix.shape[2], device=student_conf_matrix.device) * scale
            teacher_conf_matrix_scaled = torch.index_select(teacher_conf_matrix, 1, i_ids)
            teacher_conf_matrix_scaled = torch.index_select(teacher_conf_matrix_scaled, 2, j_ids)

            # compute distillation loss
            scale = 50.0
            student_conf_matrix = student_conf_matrix * scale
            teacher_conf_matrix_scaled = teacher_conf_matrix_scaled * scale

            soft_log_probs = torch_func.log_softmax(
                torch.flatten(student_conf_matrix, start_dim=1) / self.settings.temperature, dim=1)

            soft_log_targets = torch_func.log_softmax(
                torch.flatten(teacher_conf_matrix_scaled, start_dim=1) / self.settings.temperature, dim=1)

            distillation_loss = torch_func.kl_div(soft_log_probs, soft_log_targets, log_target=True, reduction='batchmean')

            distillation_loss = distillation_loss * self.settings.temperature ** 2

        # compute student loss - cross entropy
        conf_matrix_gt = conf_matrix_gt.squeeze(1)
        pos_mask = conf_matrix_gt == 1
        neg_mask = conf_matrix_gt == 0
        conf = torch.clamp(student_conf_matrix, 1e-6, 1 - 1e-6)
        loss_pos = - torch.log(conf[pos_mask])
        loss_neg = - torch.log(1 - conf[neg_mask])
        student_loss = loss_pos.mean() + loss_neg.mean()

        if self.settings.write_statistics:
            self.last_image1 = image1
            self.last_image2 = image2
            if self.settings.with_teacher:
                self.last_teacher_conf_matrix = teacher_conf_matrix.detach()
            self.last_student_conf_matrix = student_conf_matrix.detach()

        if self.settings.with_teacher:
            return [student_loss * 0.5, distillation_loss * 0.5]
        else:
            return [student_loss]

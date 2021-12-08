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


class Trainer(object):
    def __init__(self, settings, dataset_path, checkpoint_path):
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

        batch_size = self.settings.batch_size // self.settings.batch_size_divider
        self.train_dataset = MVSDataset(dataset_path, (default_cfg['input_width'], default_cfg['input_height']),
                                        epoch_size=1000)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=self.settings.data_loader_num_workers)

        self.scaler = torch.cuda.amp.GradScaler()

        self.teacher_cfg = default_cfg
        self.teacher_model = LoFTR(config=self.teacher_cfg)
        self.student_cfg = make_student_config(default_cfg)
        self.student_model = LoFTR(config=self.student_cfg)

        if self.settings.cuda:
            self.teacher_model = self.teacher_model.cuda()
            self.student_model = self.student_model.cuda()

        self.create_optimizer()

        self.kl_loss = torch.nn.KLDivLoss()

    def tensor_to_image(self, image):
        frame = image[0, :, :, :].cpu().numpy()
        res_img = (frame * 255.).astype('uint8')
        res_img = np.transpose(res_img, [1, 2, 0])  # OpenCV format
        res_img = cv2.UMat(res_img)
        return res_img.get()

    def add_image_summary(self, name, image1, image2, conf_matrix, config):
        conf_matrix = conf_matrix.detach().cpu().numpy()
        mkpts0, mkpts1, mconf = get_coarse_match(conf_matrix, config['input_height'], config['input_width'],
                                                 config['resolution'][0])
        # filter only the most confident features
        n_top = 20
        indices = np.argsort(mconf)[::-1]
        indices = indices[:n_top]
        mkpts0 = mkpts0[indices, :]
        mkpts1 = mkpts1[indices, :]

        img_size = (config['input_height'], config['input_width'])
        image1 = self.tensor_to_image(image1)
        draw_features(image1, mkpts0, img_size)
        image2 = self.tensor_to_image(image2)
        draw_features(image2, mkpts1, img_size)

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

                    # save statistics
                    if self.settings.write_statistics:
                        self.write_batch_statistics(real_batch_index)

                    if (real_batch_index + 1) % 10 == 0:
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

    def train(self, name):
        # continue training starting from the latest epoch checkpoint
        start_epoch = 0
        prev_epoch = load_last_checkpoint(self.checkpoint_path, self.student_model, self.optimizer, self.scaler)
        if prev_epoch >= 0:
            start_epoch = prev_epoch + 1
        epochs_num = start_epoch + self.epochs

        self.global_train_index = 0

        for epoch in range(start_epoch, epochs_num):
            print(f"Epoch {epoch}\n-------------------------------")
            self.student_model.train()
            train_loss = self.train_loop()
            print(f"Train Loss:{train_loss:7f} \n")
            if self.settings.write_statistics:
                self.summary_writer.add_scalar('Loss/train', train_loss, epoch)

            save_checkpoint(name, epoch, self.student_model, self.optimizer, self.scaler, self.checkpoint_path)
            self.train_dataset.reset_epoch()

    def write_batch_statistics(self, batch_index):
        if (batch_index + 1) % self.settings.statistics_period == 0:
            for name, param in self.student_model.named_parameters():
                if param.grad is not None and 'bn' not in name:
                    self.summary_writer.add_histogram(
                        tag=f"params/{name}", values=param, global_step=self.global_train_index
                    )
                    self.summary_writer.add_histogram(
                        tag=f"grads/{name}", values=param.grad, global_step=self.global_train_index
                    )

            if self.last_image1 is not None and self.last_image1 is not None:
                self.add_image_summary('teacher', self.last_image1, self.last_image2, self.last_teacher_conf_matrix,
                                       self.teacher_cfg)
                self.add_image_summary('student', self.last_image1, self.last_image2, self.last_student_conf_matrix,
                                       self.student_cfg)

    def train_loss_fn(self, image1, image2):
        if self.settings.cuda:
            image1 = image1.cuda()
            image2 = image2.cuda()

        with torch.no_grad():
            teacher_conf_matrix = self.teacher_model.forward(image1, image2)
        student_conf_matrix = self.student_model.forward(image1, image2)

        scaled_size = list(student_conf_matrix.size())[1:]
        teacher_conf_matrix_scaled = torch_func.interpolate(teacher_conf_matrix.unsqueeze(0), size=scaled_size)
        teacher_conf_matrix_scaled = teacher_conf_matrix_scaled.squeeze(0)

        target = torch_func.softmax(torch.flatten(teacher_conf_matrix_scaled))
        input = torch_func.log_softmax(torch.flatten(student_conf_matrix))
        loss_value = self.kl_loss(input, target)

        if self.settings.write_statistics:
            self.last_image1 = image1
            self.last_image2 = image2
            self.last_teacher_conf_matrix = teacher_conf_matrix
            self.last_student_conf_matrix = student_conf_matrix

        return [loss_value]

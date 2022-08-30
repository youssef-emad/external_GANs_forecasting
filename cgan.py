import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from eval_metrics import metrics_mapper
from utils import saving_criteria_satisfied


class CGAN(torch.nn.Module):
    """
    Conditional Generative Adversarial Network
    Source: https://arxiv.org/abs/1411.1784
    """

    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        device: torch.device,
    ):
        """
        Initializes a CGAN object

        Args:
            generator (torch.nn.module): Model used as generator.
                It's expected to take input_window (condition) and noise as inputs and produce
                    forecasted window as output of (num_samples, output_window_size)
            discriminator (torch.nn.module): Model used as discriminator.
                It's expected to take input_window (condition) and forecasted window as inputs and
                produce a decision wether this whole signal is true or fake
            device (torch.device): determines the device used for training wether it's cpu or gpu.
        """
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.device = device

    def train(
        self,
        training_dataloader: torch.utils.data.DataLoader,
        validation_dataloader: torch.utils.data.DataLoader,
        training_params: dict,
        output_dir: str,
    ):
        """
        Trains an instance of CGAN

        Args:
            training_dataloader (torch.utils.data.DataLoader): torch's dataloader used for training set
            validation_dataloader (torch.utils.data.DataLoader): torch's dataloader used for validation set
                (only used to calculate metrics on unseen test data and not used for gradient update)
            training_params (dict): dictionary holds the hyper-parameters required for training
                and expected to have the following format:
                {
                    "training": {
                        "num_steps": <num_training_steps>,
                        "num_steps_model_saving": <num_steps_to_regularly_save_generator>,
                    },
                    "validation": {
                        "num_steps_log": <num_steps_between_two_successive_evaluations_on_validation>,
                        "num_batches": <num_batches_used_for_validation>,
                        "num_steps_per_prediction": <num_generated_samples>,
                    },
                    "generator": {
                        "optimizer": <torch_optimizer>,
                        "loss": <torch_loss_function>,
                        "loss_type": <lorenz_or_other>,
                        "steps": <num_steps_for_generator_training>,
                        "noise_window_size": <noise_window_size_fed_to_generator>,
                    },
                    "discriminator": {
                        "optimizer": <torch_optimizer>,
                        "loss": <torch_loss_function>,
                        "steps": <num_steps_for_discriminator_training>,
                    },
                    "metrics": {
                        "training": {"generator": <list_of_metrics_names>, "discriminator": <list_of_metrics_names>},
                        "validation": <list_of_metrics_names>,
                        "best_saving_criteria": {"metric": <metric_name>, "mode": <min_or_max>},
                    }
                }
            output_dir (str): path to output directory used for model saving and tensorboard logging
        """

        # create iterators for dataloaders
        training_iter = iter(training_dataloader)

        num_steps_training = training_params["training"]["num_steps"]
        num_steps_saving = training_params["training"]["num_steps_model_saving"]

        num_steps_val_log = training_params["validation"]["num_steps_log"]
        num_batches_val = training_params["validation"]["num_batches"]
        num_steps_per_pred = training_params["validation"]["num_steps_per_prediction"]

        # load discriminator parameters
        dis_steps = training_params["discriminator"]["steps"]
        dis_optimizer = training_params["discriminator"]["optimizer"]
        dis_loss = training_params["discriminator"]["loss"]

        # load generator parameters
        gen_steps = training_params["generator"]["steps"]
        gen_optimizer = training_params["generator"]["optimizer"]
        gen_loss = training_params["generator"]["loss"]
        gen_loss_type = training_params["generator"]["loss_type"]
        noise_size = training_params["generator"]["noise_window_size"]

        # load evaluation metrics
        gen_metrics = training_params["metrics"]["training"]["generator"]
        dis_metrics = training_params["metrics"]["training"]["discriminator"]
        val_metrics = training_params["metrics"]["validation"]
        best_saving_criteria = training_params["metrics"]["best_saving_criteria"]

        # initialize the optimizers and loss functions
        dis_optimizer = dis_optimizer(self.discriminator.parameters())
        gen_optimizer = gen_optimizer(self.generator.parameters())

        gen_loss = gen_loss().to(self.device)
        dis_loss = dis_loss().to(self.device)

        tensorboard_dir = os.path.join(output_dir, "tensorboard")
        tb_writer = SummaryWriter(log_dir=tensorboard_dir)

        curr_best = np.inf

        steps_range = trange(num_steps_training, desc="Loss", leave=True)

        # in each training step, the discriminator is firstly trained for dis_steps then followed by
        # training the generator for gen_steps
        for step in steps_range:
            step_metrics = defaultdict(int)
            d_loss = 0

            for _ in range(dis_steps):

                # a StopIteration exception is raised when reaching the end of the dataloader
                # so we need to reset it by re-assigning the iterator
                try:
                    batch = next(training_iter)
                    x_train, y_train, *_ = batch
                except StopIteration:
                    training_iter = iter(training_dataloader)
                    batch = next(training_iter)
                    x_train, y_train, *_ = batch

                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)

                self.discriminator.zero_grad()
                dis_output_real = self.discriminator(x_train, y_train)
                dis_gt_real = torch.full_like(dis_output_real, 1, device=self.device)

                dis_loss_real = dis_loss(dis_output_real, dis_gt_real)
                dis_loss_real.backward()

                d_loss += dis_loss_real.detach().cpu().numpy()

                noise_batch = torch.normal(
                    0,
                    1,
                    (x_train.shape[0], noise_size),
                    device=self.device,
                    dtype=torch.float32,
                )

                y_fake = self.generator(x_train, noise_batch).detach()
                dis_output_fake = self.discriminator(x_train, y_fake)
                dis_gt_fake = torch.full_like(dis_output_fake, 0, device=self.device)

                dis_loss_fake = dis_loss(dis_output_fake, dis_gt_fake)
                dis_loss_fake.backward()

                dis_optimizer.step()
                d_loss += dis_loss_fake.detach().cpu().numpy()

                # compute discriminator metrics
                for metric_name in dis_metrics:
                    metric_func = metrics_mapper[metric_name]
                    y_pred = (
                        torch.cat([dis_output_real, dis_output_fake])
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    y_true = (
                        torch.cat([dis_gt_real, dis_gt_fake]).detach().cpu().numpy()
                    )
                    step_metrics[f"train/discriminator/{metric_name}"] += (
                        metric_func(y_true, y_pred) / dis_steps
                    )
            d_loss = 0 if not dis_steps else d_loss / dis_steps

            # train generator
            g_loss = 0
            for _ in range(gen_steps):
                try:
                    batch = next(training_iter)
                    x_train, y_train, *_ = batch
                except StopIteration:
                    training_iter = iter(training_dataloader)
                    batch = next(training_iter)
                    x_train, y_train, *_ = batch

                self.generator.zero_grad()
                noise_batch = torch.normal(
                    0,
                    1,
                    (x_train.shape[0], noise_size),
                    device=self.device,
                    dtype=torch.float32,
                )

                x_train = x_train.to(self.device)
                y_fake = self.generator(x_train, noise_batch)
                dis_output_fake = self.discriminator(x_train, y_fake)

                # Mackey-Glass works best with Minmax loss in our expriements while other dataset
                # produce their best result with non-saturated loss
                if gen_loss_type == "lorenz":
                    gen_loss_fake = gen_loss(
                        dis_output_fake,
                        torch.full_like(dis_output_fake, 1, device=self.device),
                    )
                else:
                    gen_loss_fake = -1 * gen_loss(
                        dis_output_fake,
                        torch.full_like(dis_output_fake, 0, device=self.device),
                    )
                gen_loss_fake.backward()
                gen_optimizer.step()

                g_loss += gen_loss_fake.detach().cpu().numpy()

                # compute generator metrics
                for metric_name in gen_metrics:
                    metric_func = metrics_mapper[metric_name]
                    step_metrics[f"train/generator/{metric_name}"] += (
                        metric_func(
                            y_train.detach().cpu().numpy(),
                            y_fake.detach().cpu().numpy(),
                        )
                        / gen_steps
                    )

            g_loss = 0 if not gen_steps else g_loss / gen_steps

            # write training losses to tensorboard
            tb_writer.add_scalar("train/discriminator/loss", d_loss, step)
            tb_writer.add_scalar("train/generator/loss", g_loss, step)

            # regular model saving
            if step % num_steps_saving == 0:
                torch.save(
                    self.generator,
                    os.path.join(
                        output_dir,
                        "checkpoints",
                        f"step_{step}_{metric_name.replace('/', '_')}_{curr_best:.4f}.pth",
                    ),
                )

            # validation
            if step % num_steps_val_log == 0:

                validation_iter = iter(validation_dataloader)

                gt_val = []
                preds_val = []

                for _ in range(num_batches_val):

                    try:
                        batch = next(validation_iter)
                        x_val, y_val, *_ = batch
                    except StopIteration:
                        validation_iter = iter(validation_dataloader)
                        batch = next(validation_iter)
                        x_val, y_val, *_ = batch

                    x_val = x_val.to(self.device)
                    gt_val.extend(y_val.detach().cpu().numpy())

                    y_pred = []
                    for _ in range(num_steps_per_pred):
                        noise_batch = torch.normal(
                            0,
                            1,
                            (x_val.shape[0], noise_size),
                            device=self.device,
                            dtype=torch.float32,
                        )
                        y_pred.append(
                            self.generator(x_val, noise_batch).detach().cpu().numpy()
                        )

                    y_pred = np.array(y_pred).transpose(1, 0, 2)
                    preds_val.extend(y_pred)

                # compute validation metrics
                for metric_name in val_metrics:
                    metric_func = metrics_mapper[metric_name]
                    step_metrics[f"validation/{metric_name}"] += metric_func(
                        np.array(gt_val), np.array(preds_val)
                    )

            # write to tensorboard
            for key, value in step_metrics.items():
                tb_writer.add_scalar(key, value, step)

            # save generator and update current best
            if saving_criteria_satisfied(
                step_metrics=step_metrics,
                saving_criteria=best_saving_criteria,
                current_best=curr_best,
            ):
                metric_name = best_saving_criteria["metric"]
                curr_best = step_metrics[metric_name]
                torch.save(
                    self.generator,
                    os.path.join(
                        output_dir,
                        "checkpoints",
                        f"best_model_step_{step}_{metric_name.replace('/', '_')}_{curr_best:.4f}.pth",
                    ),
                )
                print(f"Current Best updated {curr_best:.4f} at step {step}")

            # update progress bar
            steps_range.set_description(f"D: {d_loss:.4f}, G: {g_loss:.4f}")
        tb_writer.flush()

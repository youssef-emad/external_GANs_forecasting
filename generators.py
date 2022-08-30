import os
import numpy as np
import torch
from tqdm import trange
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


from eval_metrics import metrics_mapper
from utils import saving_criteria_satisfied


class G_Base(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def train(
        self,
        training_dataloader: torch.utils.data.DataLoader,
        validation_dataloader: torch.utils.data.DataLoader,
        training_params: dict,
        output_dir: str,
        device: torch.device,
    ):
        """
        Trains an instance of G_Base (time-series forecasting model)

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
                    },
                    "generator": {
                        "optimizer": <torch_optimizer>,
                        "loss": <torch_loss_function>,
                    },
                    "metrics": {
                        "training": {"generator": <list_of_metrics_names>},
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

        # load generator parameters
        optimizer = training_params["generator"]["optimizer"]
        loss_func = training_params["generator"]["loss"]

        # load evaluation metrics
        train_metrics = training_params["metrics"]["training"]["generator"]
        val_metrics = training_params["metrics"]["validation"]
        best_saving_criteria = training_params["metrics"]["best_saving_criteria"]

        optimizer = optimizer(self.parameters())
        loss_func = loss_func().to(device)

        tensorboard_dir = os.path.join(output_dir, "tensorboard")
        tb_writer = SummaryWriter(log_dir=tensorboard_dir)

        curr_best = np.inf

        steps_range = trange(num_steps_training, desc="Loss", leave=True)

        for step in steps_range:
            step_metrics = defaultdict(int)

            try:
                batch = next(training_iter)
                x_train, y_train, *_ = batch
            except StopIteration:
                training_iter = iter(training_dataloader)
                batch = next(training_iter)
                x_train, y_train, *_ = batch

            self.zero_grad()

            y_pred = self.forward(x_train)

            loss = loss_func(y_train, y_pred)

            loss.backward()
            optimizer.step()

            # compute training metrics
            for metric_name in train_metrics:
                metric_func = metrics_mapper[metric_name]
                step_metrics[f"train/generator/{metric_name}"] = metric_func(
                    y_train.detach().cpu().numpy(),
                    y_pred.detach().cpu().numpy(),
                )

            loss_value = loss.detach().cpu().numpy()

            # write training loss to tensorboard
            tb_writer.add_scalar("train/stand-alone/loss", loss_value, step)

            # regular model saving
            if step % num_steps_saving == 0:
                torch.save(
                    self,
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

                    gt_val.extend(y_val.detach().cpu().numpy())

                    preds_val.extend(self(x_val).detach().cpu().numpy())

                # compute validation metrics
                for metric_name in val_metrics:
                    metric_func = metrics_mapper[metric_name]
                    step_metrics[f"validation/{metric_name}"] += metric_func(
                        np.array(gt_val), np.array(preds_val)
                    )

            # write to tensorboard
            for key, value in step_metrics.items():
                tb_writer.add_scalar(key, value, step)

            # save model and update current best
            if saving_criteria_satisfied(
                step_metrics=step_metrics,
                saving_criteria=best_saving_criteria,
                current_best=curr_best,
            ):
                metric_name = best_saving_criteria["metric"]
                curr_best = step_metrics[metric_name]
                torch.save(
                    self,
                    os.path.join(
                        output_dir,
                        "checkpoints",
                        f"best_model_step_{step}_{metric_name.replace('/', '_')}_{curr_best:.4f}.pth",
                    ),
                )
                print(f"Current Best updated {curr_best:.4f} at step {step}")

            # update progress bar
            steps_range.set_description(f"G: {loss_value:.4f}")
        tb_writer.flush()


class G_CNN_RNN(G_Base):
    def __init__(
        self,
        out_window_size,
        noise_size,
        condition_size,
        latent_size,
        rnn_cell_type,
        n_filters,
        kernel_size,
        pool_size,
        mean=0,
        std=1,
    ):
        super().__init__()

        self.out_window_size = out_window_size
        self.noise_size = noise_size
        self.condition_size = condition_size
        self.latent_size = latent_size
        self.mean = mean
        self.std = std

        self.CNN = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1, out_channels=n_filters, kernel_size=kernel_size
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=n_filters,
                out_channels=2 * n_filters,
                kernel_size=kernel_size,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(pool_size),
            torch.nn.Flatten(),
            torch.nn.Linear(
                int(
                    2
                    * n_filters
                    * ((self.condition_size - 2 * (kernel_size - 1)) / pool_size)
                ),
                out_features=latent_size,
            ),
        )

        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=latent_size + self.noise_size,
                out_features=latent_size + self.noise_size,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=latent_size + self.noise_size,
                out_features=self.out_window_size,
            ),
        )

        if rnn_cell_type.lower() == "lstm":
            self.cond_to_latent_lstm = torch.nn.LSTM(
                input_size=1, hidden_size=latent_size
            )
        elif rnn_cell_type.lower() == "gru":
            self.cond_to_latent_lstm = torch.nn.GRU(
                input_size=1, hidden_size=latent_size
            )
        else:
            raise KeyError(
                f"Unknown RNN cell type {rnn_cell_type}. Only LSTM and GRU are available"
            )

    def forward(self, condition, noise=None):

        condition = (condition - self.mean) / self.std
        condition = condition.view(-1, 1, self.condition_size)
        cnn_output = self.CNN(condition)

        lstm_input = cnn_output.view(-1, self.latent_size, 1)
        lstm_input = lstm_input.transpose(0, 1)

        lstm_output, _ = self.cond_to_latent_lstm(lstm_input)
        lstm_output = lstm_output[-1]

        if noise is not None:
            lstm_output = torch.cat((lstm_output, noise), dim=1)

        output = self.MLP(lstm_output)
        output = output * self.std + self.mean

        return output

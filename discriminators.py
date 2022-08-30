import torch


class D_CNN(torch.nn.Module):
    def __init__(
        self,
        total_sequence_length: int,
        condition_size: int,
        latent_size: int,
        n_filters: int,
        kernel_size: int,
        pool_size: int,
        mean=0,
        std=1,
    ):
        super().__init__()
        self.total_sequence_length = total_sequence_length
        self.latent_size = latent_size
        self.condition_size = condition_size
        self.mean = mean
        self.std = std

        self.model = torch.nn.Sequential(
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
                    * (
                        (self.total_sequence_length - 2 * (kernel_size - 1))
                        // pool_size
                    )
                ),
                out_features=latent_size,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=latent_size, out_features=1),
            torch.nn.Sigmoid(),
        )

    def forward(
        self, condition: torch.tensor, prediction: torch.tensor
    ) -> torch.tensor:
        d_input = torch.cat(
            (
                condition,
                prediction.view(-1, self.total_sequence_length - self.condition_size),
            ),
            dim=1,
        )
        d_input = (d_input - self.mean) / self.std
        d_input = d_input.view(-1, 1, self.total_sequence_length)
        output = self.model(d_input)
        return output


class D_RNN(torch.nn.Module):
    def __init__(
        self,
        total_sequence_length: int,
        condition_size: int,
        latent_size: int,
        rnn_cell_type: str,
        mean=0,
        std=1,
    ):
        super().__init__()
        self.total_sequence_length = total_sequence_length
        self.latent_size = latent_size
        self.condition_size = condition_size
        self.mean = mean
        self.std = std

        if rnn_cell_type.lower() == "lstm":
            self.input_to_latent = torch.nn.LSTM(
                input_size=1, hidden_size=self.latent_size
            )
        elif rnn_cell_type.lower() == "gru":
            self.input_to_latent = torch.nn.GRU(
                input_size=1, hidden_size=self.latent_size
            )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.latent_size, out_features=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, cd, prediction):
        d_input = torch.cat(
            (
                cd,
                prediction.view(-1, self.total_sequence_length - self.condition_size),
            ),
            dim=1,
        )
        d_input = (d_input - self.mean) / self.std
        d_input = d_input.view(-1, self.total_sequence_length, 1)
        d_input = d_input.transpose(0, 1)
        d_latent, _ = self.input_to_latent(d_input)
        d_latent = d_latent[-1]
        output = self.model(d_latent)
        return output

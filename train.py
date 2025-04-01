import torch
from utils import get_data
from torch.utils.data import TensorDataset, DataLoader

encoder_length, decoder_length, num_test_samples = 3, 3, 100


def create_sliding_windows(timeseries_list, encoder_history, forecast_horizon):
    windowed_data = []  # Store extracted windows

    for series in timeseries_list:
        series_length = len(series)
        windows = []

        for i in range(series_length - encoder_history - forecast_horizon + 1):
            x = series[i : i + encoder_history]  # Input history
            y = series[
                i + encoder_history : i + encoder_history + forecast_horizon
            ]  # Future values
            windows.append((x, y))

        windowed_data.append(windows)

    return windowed_data


data = get_data()
number_of_goals = len(data["goal_status"].unique())
data_lists = data.groupby("id")["goal_status"].agg(list).tolist()


x = [torch.tensor(sample) for sample in data_lists]

x_windowed = create_sliding_windows(x, encoder_length, decoder_length)
source, target = torch.stack(
    [pair[0] for sample in x_windowed for pair in sample]
), torch.stack([pair[1] for sample in x_windowed for pair in sample])

source_train, target_train, source_test, target_test = (
    source[:-num_test_samples],
    target[:-num_test_samples],
    source[-num_test_samples:],
    target[-num_test_samples:],
)

train_dataset, test_dataset = TensorDataset(source_train, target_train), TensorDataset(
    source_test, target_test
)

train_dataloader, test_dataloader = DataLoader(
    train_dataset, batch_size=16, shuffle=True
), DataLoader(test_dataset, batch_size=16, shuffle=True)

encoder, decoder, projection = (
    torch.nn.GRU(number_of_goals, 128, batch_first=True),
    torch.nn.GRU(number_of_goals, 128, batch_first=True),
    torch.nn.Linear(128, number_of_goals),
)

optimizer = torch.optim.Adam(
    list(encoder.parameters())
    + list(decoder.parameters())
    + list(projection.parameters()),
    lr=0.001,
)


def forward_data(train: bool = True) -> float:
    dataloader = train_dataloader if train else test_dataloader
    losses = []

    for source, target in dataloader:
        if train:
            optimizer.zero_grad()
        _, encoder_repr = encoder(
            torch.nn.functional.one_hot(source, num_classes=number_of_goals).float()
        )

        decoder_input = torch.cat(
            (
                torch.zeros(source.shape[0], 1, number_of_goals),
                torch.nn.functional.one_hot(
                    target[:, :-1], num_classes=number_of_goals
                ),
            ),
            dim=1,
        )

        decoder_hidden, _ = decoder(decoder_input, encoder_repr)
        predictions = projection(decoder_hidden)
        loss = torch.nn.functional.cross_entropy(predictions.permute(0, 2, 1), target)
        if train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

    return torch.tensor(losses).mean().item()


for epoch_i in range(200):
    train_loss, test_loss = forward_data(), forward_data(train=False)
    print(
        "epoch: {}, train loss: {:.2f}, test loss: {:.2f}".format(
            epoch_i + 1, train_loss, test_loss
        )
    )

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # Maximum valid starting index: we need context_length tokens for input
    # plus one more token for the last label (shifted by 1)
    max_start = len(dataset) - context_length - 1

    # Randomly sample batch_size starting positions
    start_indices = torch.randint(0, max_start + 1, (batch_size,))

    # Build input (x) and label (y) tensors
    # x[i] = dataset[start : start + context_length]
    # y[i] = dataset[start+1 : start+1 + context_length]  (next-token prediction)
    x = torch.stack([torch.from_numpy(dataset[i : i + context_length].copy()) for i in start_indices])
    y = torch.stack([torch.from_numpy(dataset[i + 1 : i + 1 + context_length].copy()) for i in start_indices])

    # Ensure LongTensor type and move to the specified device
    return x.long().to(device), y.long().to(device)
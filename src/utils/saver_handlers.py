import os
import torch
import numpy as np

class SaverHandler:
    """
    Modular handler for saving model outputs (tensors) during testing or validation.

    This class provides a unified interface for saving intermediate or final
    results produced by a model (e.g., reconstructions, anomaly maps, or inputs).
    It supports two flexible modes:
      1. Saving all tensors every N batches.
      2. Saving tensors corresponding to specific dataset indices.

    Parameters
    ----------
    output_dir : str
        Base directory where tensors will be saved.
    save_every : int, optional
        Frequency (in batches) at which to save all tensors.
        Mutually exclusive with `save_indices`.
    save_indices : list of int, optional
        List of global dataset indices whose tensors should be saved.
        Mutually exclusive with `save_every`.
    name_map : dict, optional
        Optional mapping from `output_dict` keys to output filenames.

    Attributes
    ----------
    output_dir : str
        Directory where saved tensors are written.
    save_every : int or None
        If set, determines periodic saving frequency by batch index.
    save_indices : list of int
        List of dataset indices to save if index-based saving is used.
    name_map : dict
        Maps output_dict keys to filenames.
    _saved : set
        Tracks which dataset indices have already been saved.

    Methods
    -------
    save_batch(batch_idx, batch_size, output_dict, sample_indices)
        Save tensors for the current batch based on either batch frequency
        or global dataset indices.

    Example
    -------
    >>> saver = SaverHandler(output_dir="results/tensors", save_every=10)
    >>> saver.save_batch(batch_idx, batch_size, output_dict, sample_indices)

    >>> saver = SaverHandler(output_dir="results/tensors", save_indices=[0, 2, 18])
    >>> saver.save_batch(batch_idx, batch_size, output_dict, sample_indices)
    """

    def __init__(
        self,
        output_dir: str,
        save_every: int = None,
        save_indices: list[int] = None,
        name_map: dict[str, str] = None,
    ):
        """
        Parameters
        ----------
        output_dir : str
            Base directory where results will be saved.
        save_every : int, optional
            Save every N batches. Mutually exclusive with save_indices.
        save_indices : list[int], optional
            Global dataset indices to save.
        name_map : dict, optional
            Optional mapping from output_dict keys to filenames.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.save_every = save_every
        self.save_indices = save_indices or []
        self.name_map = name_map or {}
        self._saved = set()  # keep track of which indices were already saved

    def _save_tensor(self, tensor: torch.Tensor, filename: str):
        path = os.path.join(self.output_dir, filename)
        torch.save(tensor.detach().cpu(), path)

    def save_batch(
        self,
        batch_idx: int,
        batch_size: int,
        output_dict: dict,
        sample_indices: list[int],
    ):
        """
        Save tensors based on either batch frequency or specific dataset indices.
        """
        # Mode 1: save every N batches
        if self.save_every is not None:
            if batch_idx % self.save_every == 0:
                batch_dir = os.path.join(self.output_dir, f"batch_{batch_idx:03d}")
                os.makedirs(batch_dir, exist_ok=True)
                for key, tensor in output_dict.items():
                    if tensor is None:
                        continue
                    if isinstance(tensor, np.ndarray):
                        tensor = torch.from_numpy(tensor)
                    filename = self.name_map.get(key, key) + ".pt"
                    self._save_tensor(tensor, os.path.join(batch_dir, filename))
            return

        # Mode 2: save specific dataset indices
        if self.save_indices:
            for global_idx, local_pos in zip(
                sample_indices, range(len(sample_indices))
            ):
                if global_idx in self.save_indices and global_idx not in self._saved:
                    self._saved.add(global_idx)
                    item_dir = os.path.join(self.output_dir, f"sample_{global_idx:05d}")
                    os.makedirs(item_dir, exist_ok=True)
                    for key, tensor in output_dict.items():
                        if tensor is None:
                            continue
                        if isinstance(tensor, np.ndarray):
                            tensor = torch.from_numpy(tensor)
                        tensor_to_save = tensor[local_pos].unsqueeze(0)
                        filename = self.name_map.get(key, key) + ".pt"
                        self._save_tensor(
                            tensor_to_save, os.path.join(item_dir, filename)
                        )

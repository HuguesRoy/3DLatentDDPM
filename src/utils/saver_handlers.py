import os
import torch
import numpy as np
from typing import Dict, List, Optional


class SaverHandler:
    """
    Modular handler for saving model outputs (tensors/arrays) during testing or validation.

    Supports:
      1. Saving all tensors every N batches.
      2. Saving tensors corresponding to specific dataset indices.
      3. Saving format: .pt (torch), .npy (numpy), or both.

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
        Optional mapping from `output_dict` keys to output base filenames
        (without extension).
    save_format : {"pt", "npy", "both"}, optional
        Storage format. Default: "pt".
    """

    def __init__(
        self,
        output_dir: str,
        save_every: Optional[int] = None,
        save_indices: Optional[List[int]] = None,
        name_map: Optional[Dict[str, str]] = None,
        save_format: str = "pt",
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        if save_every is not None and save_indices:
            raise ValueError("`save_every` and `save_indices` are mutually exclusive.")

        if save_format not in {"pt", "npy", "both"}:
            raise ValueError("save_format must be one of {'pt', 'npy', 'both'}")

        self.save_every = save_every
        self.save_indices = save_indices or []
        self.name_map = name_map or {}
        self.save_format = save_format
        self._saved = set()  # keep track of which indices were already saved

    # ---- low-level save helpers ----

    def _save_pt(self, tensor: torch.Tensor, path_root: str):
        """
        Save tensor as a .pt file.
        path_root: full path without extension.
        """
        path = path_root + ".pt"
        torch.save(tensor.detach().cpu(), path)

    def _save_npy(self, array: np.ndarray, path_root: str):
        """
        Save array as a .npy file.
        path_root: full path without extension.
        """
        path = path_root + ".npy"
        np.save(path, array)

    def _save_item(self, tensor_or_array, path_root: str):
        """
        Save one item (tensor or numpy array) according to save_format.
        path_root: full path without extension.
        """
        # Normalize to torch + numpy forms once
        if isinstance(tensor_or_array, np.ndarray):
            arr = tensor_or_array
            tensor = torch.from_numpy(arr)
        else:
            tensor = tensor_or_array.detach().cpu()
            arr = tensor.numpy()

        if self.save_format in {"pt", "both"}:
            self._save_pt(tensor, path_root)
        if self.save_format in {"npy", "both"}:
            self._save_npy(arr, path_root)

    # ---- public API ----

    def save_batch(
        self,
        batch_idx: int,
        batch_size: int,
        output_dict: Dict[str, torch.Tensor],
        sample_indices: List[int],
    ):
        """
        Save tensors based on either batch frequency or specific dataset indices.

        Parameters
        ----------
        batch_idx : int
            Index of the current batch.
        batch_size : int
            Number of samples in the current batch.
        output_dict : dict
            Dict of model outputs: {key: tensor or ndarray}.
            Expected shapes: (B, ...) for per-sample outputs.
        sample_indices : list[int]
            Global dataset indices corresponding to each sample in the batch.
        """
        # Mode 1: save every N batches (save the full batch)
        if self.save_every is not None:
            if batch_idx % self.save_every == 0:
                batch_dir = os.path.join(self.output_dir, f"batch_{batch_idx:03d}")
                os.makedirs(batch_dir, exist_ok=True)

                for key, tensor in output_dict.items():
                    if tensor is None:
                        continue

                    base_name = self.name_map.get(key, key)
                    path_root = os.path.join(batch_dir, base_name)
                    self._save_item(tensor, path_root)
            return

        # Mode 2: save specific dataset indices (per-sample)
        if self.save_indices:
            # local_pos is 0..B-1
            for local_pos, global_idx in enumerate(sample_indices):
                if global_idx in self.save_indices and global_idx not in self._saved:
                    self._saved.add(global_idx)
                    item_dir = os.path.join(self.output_dir, f"sample_{global_idx:05d}")
                    os.makedirs(item_dir, exist_ok=True)

                    for key, tensor in output_dict.items():
                        if tensor is None:
                            continue

                        # Take only this sample: shape (1, ...)
                        if isinstance(tensor, np.ndarray):
                            sample_item = tensor[local_pos : local_pos + 1]
                        else:
                            sample_item = tensor[local_pos].unsqueeze(0)

                        base_name = self.name_map.get(key, key)
                        path_root = os.path.join(item_dir, base_name)
                        self._save_item(sample_item, path_root)

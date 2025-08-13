import math
import random
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import Dataset, Sampler


class ClimateDataset(Dataset):
    def __init__(
        self,
        X,
        k,
        T,
        dNEE,
        bNEE,
        dT,
        NEE,
        site_ids: Optional[Sequence[Any]] = None,
    ):
        """
        Args:
            X (numpy array): Input data of shape (n_samples, input_dim)
            k (numpy array): Ground truth for E0 and rb, shape (n_samples, 2)
            T (numpy array): Ground truth for Tair, shape (n_samples, 1)
            dNEE (numpy array): Ground truth for f, shape (n_samples, 1)
            bNEE (numpy array): Ground truth for NEE at current t (boundary condition), shape (n_samples, 1)
            dT (numpy array): Ground truth for temperature derivative, shape (n_samples, 1)
            NEE (numpy array): Ground truth for NEE at t + 1, shape (n_samples, 1)
            site_ids (optional Sequence[Any]): Site identifier for each sample (e.g., site code or int label).
                When provided, you can build a balanced batch sampler to draw roughly equal samples per site per batch.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.k = torch.tensor(k, dtype=torch.float32)
        self.T = torch.tensor(T, dtype=torch.float32)
        self.dNEE = torch.tensor(dNEE, dtype=torch.float32)
        self.bNEE = torch.tensor(bNEE, dtype=torch.float32)
        self.dT = torch.tensor(dT, dtype=torch.float32)
        self.NEE = torch.tensor(NEE, dtype=torch.float32)

        if site_ids is not None:
            if len(site_ids) != len(self.X):
                raise ValueError("Length of site_ids must match number of samples")
            # Keep as a Python list for flexible identifiers (str/int)
            self.site_ids: Optional[List[Any]] = list(site_ids)
        else:
            self.site_ids = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Get current NEE and previous NEE for boundary condition
        sample: Dict[str, torch.Tensor] = {
            'X': self.X[idx],
            'k': self.k[idx],
            'T': self.T[idx],
            'dNEE': self.dNEE[idx],
            'bNEE': self.bNEE[idx],
            'dT': self.dT[idx],
            'NEE': self.NEE[idx]
        }
        # Optionally include site identifier (as-is, not a tensor), if available
        if self.site_ids is not None:
            # Note: this is intentionally not a torch.Tensor to preserve original type (e.g., string site code)
            sample['site_id'] = self.site_ids[idx]  # type: ignore[index]
        return sample

    def build_balanced_batch_sampler(
        self,
        batch_size: int,
        replacement: bool = True,
        drop_last: bool = False,
        num_batches: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> "BalancedBatchSampler":
        """Create a sampler that yields batches with approximately equal counts per site.

        Use with DataLoader as: DataLoader(dataset, batch_sampler=dataset.build_balanced_batch_sampler(...))
        """
        if self.site_ids is None:
            raise ValueError("site_ids must be provided to use balanced batch sampling")
        return BalancedBatchSampler(
            site_ids=self.site_ids,
            dataset_size=len(self),
            batch_size=batch_size,
            replacement=replacement,
            drop_last=drop_last,
            num_batches=num_batches,
            generator=generator,
        )


class BalancedBatchSampler(Sampler[List[int]]):
    """Round-robin balanced batch sampler by site.

    Ensures each batch draws from sites as evenly as possible by cycling through sites.
    If batch_size is not divisible by the number of sites, some sites will appear one time
    more than others in a batch (difference at most 1).

    By default, samples with replacement, so all sites contribute on every batch even if
    some sites have fewer samples.
    """

    def __init__(
        self,
        site_ids: Sequence[Any],
        dataset_size: int,
        batch_size: int,
        replacement: bool = True,
        drop_last: bool = False,
        num_batches: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if len(site_ids) != dataset_size:
            raise ValueError("site_ids length must equal dataset_size")

        self.batch_size = batch_size
        self.replacement = replacement
        self.drop_last = drop_last
        self.num_batches = num_batches
        self.generator = generator

        # Build per-site index pools
        site_to_indices: Dict[Any, List[int]] = {}
        for index, site in enumerate(site_ids):
            site_to_indices.setdefault(site, []).append(index)

        self.sites: List[Any] = list(site_to_indices.keys())
        if len(self.sites) == 0:
            raise ValueError("At least one site is required")
        self._site_pools: Dict[Any, List[int]] = site_to_indices
        # Initialize shuffles
        self._reset_all_pools()

        # Precompute round-robin order of sites per batch position for efficiency
        self._round_robin_site_order: List[int] = list(range(len(self.sites)))

    def _rng(self) -> random.Random:
        if self.generator is None:
            return random
        # Derive a deterministic seed from the torch generator for python's RNG
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self.generator).item())
        return random.Random(seed)

    def _reset_all_pools(self) -> None:
        rng = self._rng()
        for site, indices in self._site_pools.items():
            rng.shuffle(indices)
        # Pointers per site into the shuffled pools
        self._site_ptrs: Dict[Any, int] = {site: 0 for site in self.sites}

    def _draw_from_site(self, site: Any) -> Optional[int]:
        indices = self._site_pools[site]
        ptr = self._site_ptrs[site]

        if ptr >= len(indices):
            if not self.replacement:
                return None
            # Reshuffle and wrap around when sampling with replacement
            self._site_ptrs[site] = 0
            rng = self._rng()
            rng.shuffle(indices)
            ptr = 0

        idx = indices[ptr]
        self._site_ptrs[site] = ptr + 1
        return idx

    def __iter__(self) -> Iterator[List[int]]:
        num_sites = len(self.sites)
        if num_sites == 0:
            return iter(())

        total_batches: Optional[int]
        if self.num_batches is not None:
            total_batches = self.num_batches
        else:
            # Default: cover the dataset once (approximately) even with replacement
            total_batches = math.ceil(sum(len(pool) for pool in self._site_pools.values()) / self.batch_size)

        batches_yielded = 0
        rng = self._rng()
        while total_batches is None or batches_yielded < total_batches:
            # Choose site order for this batch to avoid same start site every time
            site_order = self._round_robin_site_order[:]
            rng.shuffle(site_order)

            batch: List[int] = []
            for position in range(self.batch_size):
                site_index = site_order[position % num_sites]
                site = self.sites[site_index]
                idx = self._draw_from_site(site)
                if idx is None:
                    # Cannot fill batch without replacement
                    if self.drop_last:
                        break
                    else:
                        # Yield a smaller final batch
                        if batch:
                            yield batch
                        return
                batch.append(idx)

            if len(batch) == self.batch_size or (len(batch) > 0 and not self.drop_last):
                yield batch
                batches_yielded += 1
            elif len(batch) == 0:
                # Nothing else can be yielded
                return

    def __len__(self) -> int:
        if self.num_batches is not None:
            return self.num_batches
        # Approximate number of batches to cover all samples once
        total = sum(len(pool) for pool in self._site_pools.values())
        full = total // self.batch_size
        if total % self.batch_size == 0 or self.drop_last:
            return full
        return full + 1

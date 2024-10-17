# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from typing import Dict, List, Optional, Union

import torch
from torch.nn import functional as F

from torch.utils.data import Dataset
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX, PACK_TYPE
from tqdm import tqdm
from torchtune.training import get_world_size_and_rank
import pdb
PACK_TYPE = Dict[str, Union[torch.Tensor, List[int]]]

class NotPackedDataset(Dataset):
    def __init__(
        self,
        ds: Dataset,
        max_seq_len: int,
        padding_idx: int = 0,
        max_packs: Optional[int] = None,
        split_across_pack: bool = False,
    ) -> None:
        self.ds = ds
        # self.perplexity = ds['perplexity']
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.max_packs = max_packs
        self.split_across_pack = split_across_pack
        # Where final samples will be held
        self.packs: List[PACK_TYPE] = []
        self.previous_sample_boundary: int = 0
        self._pack()

    def _pack(self) -> None:
        """Iterate through the dataset. Use a buffer to hold samples until max_seq_len,
        then append the buffer to self.packs as a single "packed" sample. Continue
        until max_packs or end of dataset."""
        # Buffer to hold samples until they are long enough to be added to self.packs
        current_pack = {
            "tokens": [],
            "labels": [],
            "input_pos": [],
            "seq_lens": [],
        }

        # Only show progress bar on rank 0
        _, rank = get_world_size_and_rank()
        if rank == 0:
            pbar = tqdm(total=len(self.ds), desc="Masking dataset", dynamic_ncols=True)

        for sample in self.ds:
            tokens, labels = sample["tokens"], sample["labels"]

            # If the dataset outputs samples that are larger than the specified
            # max_seq_len and we're unable to split it, user needs to modify
            # one of the two parameters
            seq_len = len(tokens)
            # if seq_len > self.max_seq_len and not self.split_across_pack:
            #     raise ValueError(
            #         f"Dataset sample is too long ({seq_len} > {self.max_seq_len}). "
            #         "Please set `split_across_pack=True` or increase `max_seq_len`."
            #     )

            # If the current pack is over the max_seq_len, add it to self.packs and
            # retain any truncated or bumped samples for next pack
            if len(tokens) < self.max_seq_len:

                # Update the current pack
                current_pack["tokens"] = tokens
                current_pack["labels"] = labels
                current_pack["input_pos"] = list(range(seq_len))
                current_pack["seq_lens"] = [seq_len]
                
                pack = self._convert_to_tensors(current_pack)
                pack = self._pad_pack(pack=pack, padding_idx=self.padding_idx)

                self.packs.append(pack)

            elif len(tokens) > self.max_seq_len:
                tokens_big, labels_big = self.split_list(tokens=tokens, labels=labels)
                for tokens, labels in zip(tokens_big, labels_big):
                    current_pack["tokens"] = self.add_eos_bos(tokens)
                    current_pack["labels"] = self.add_eos_bos(labels)

                    seq_len = len(current_pack["tokens"])

                    current_pack["input_pos"] = list(range(seq_len))
                    current_pack["seq_lens"] = [seq_len]

                    pack = self._convert_to_tensors(current_pack)
                    pack = self._pad_pack(pack=pack, padding_idx=self.padding_idx)

                    self.packs.append(pack)

            if rank == 0:
                pbar.update()

    
    def split_list(self, tokens, labels):
        tokens = [tokens[i:i + self.max_seq_len] for i in range(0, len(tokens), self.max_seq_len)]
        labels = [labels[i:i + self.max_seq_len] for i in range(0, len(labels), self.max_seq_len)]
        return tokens, labels

    def add_eos_bos(self, data):
        data[0] = 128000
        data[-1] = 128001
        return data
    
    def casual_mask(self, seq_len, tokens, labels):
        x = [token.item() for token in tokens if token.item() != 128004]
        y = [token.item() for token in labels if token.item() != -100]

        diag = len(x) - len(y)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=diag)
        return mask==0

    def _convert_to_tensors(self, pack: PACK_TYPE) -> PACK_TYPE:
        """Converts a pack into tensors. Pack comes in as a dict of lists and is converted to tensors.
        The only key that does not get converted is ``seq_lens``.
        """
        return {
            "tokens": torch.tensor(pack["tokens"]),
            "labels": torch.tensor(pack["labels"]),
            "input_pos": torch.tensor(pack["input_pos"]),
            "seq_lens": pack["seq_lens"],
        }

            
    def _pad_pack(self, pack: PACK_TYPE, padding_idx: int) -> PACK_TYPE:
        """Pads a pack to ``self.max_seq_len``."""
        # Pad tokens
        padded_tokens = F.pad(
            pack["tokens"],
            (0, self.max_seq_len - len(pack["tokens"])),
            value=padding_idx,
        )

        # Pad labels
        padded_labels = F.pad(
            pack["labels"],
            (0, self.max_seq_len - len(pack["labels"])),
            value=CROSS_ENTROPY_IGNORE_IDX,
        )

        # Pad input_pos continuing the sequence from last value
        # in input_pos
        # e.g. [0 1 2] -> [0 1 2 3 4 5] for self.max_seq_len = 6
        num_range = torch.arange(
            pack["input_pos"][-1] + 1,
            pack["input_pos"][-1] + self.max_seq_len - len(pack["input_pos"]) + 1,
        )
        # Clamp to max_seq_len - 1 to avoid out of bounds error
        clamped_num_range = torch.clamp(num_range, 0, self.max_seq_len - 1)
        padded_input_pos = torch.cat([pack["input_pos"], clamped_num_range])

        return {
            "tokens": padded_tokens,
            "labels": padded_labels,
            "input_pos": padded_input_pos,
            "seq_lens": pack["seq_lens"],  # seq_len is untouched
        }
    
    def combine_tensors(self, pkr, m):
        # Convert the tensors to numpy arrays for easier manipulation
        pkr = np.array(pkr)
        m = np.array(m)
        
        # Initialize an empty list to hold the resulting rows
        result = []
        
        # Iterate through the pkr mask
        for i in range(len(pkr)):
            if pkr[i]:
                # If the mask is True, include the corresponding row from m
                result.append(m[i])
            else:
                # If the mask is False, include a row of False values
                result.append([False] * m.shape[1])
        
        return np.array(result)
    
    def __len__(self) -> int:
        return len(self.packs)
    

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Constructs the attention mask on-the-fly and returns whole sample."""
        current_pack = self.packs[idx]

        # pre_mask = current_pack['tokens'] != self.padding_idx
        # mask =  self.casual_mask(self.max_seq_len)[:pre_mask.sum(), :]

        # mini_false = self.max_seq_len - pre_mask.sum()
        # mini_t = torch.zeros(mini_false, self.max_seq_len) == 1
        
        # casual_mask = torch.cat((mask, mini_t), 0)
        
        casual_mask = (current_pack['tokens'] != self.padding_idx) & (self.casual_mask(self.max_seq_len, current_pack['tokens'], current_pack['labels']))

        return {
            "tokens": current_pack["tokens"],
            "labels": current_pack["labels"],
            "input_pos": current_pack["input_pos"],
            # Assemble the mask into a block causal matrix
            "mask": casual_mask
        }
    
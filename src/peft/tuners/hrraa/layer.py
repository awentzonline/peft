# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import peft.tuners.hrraa.hrr as hrr
from .config import TRANSFORMERS_MODEL_CONFIG


class HRRAdaptedAttention(nn.Module):
    """This module wraps a LLamaAttention module and injects adaption prompts."""

    def __init__(self, model_type: str, model):
        """
        Initialize object.

        Args:
            model_type: The transformer model type. This is used to retrieve the right method to
                compute query states.
            model: The original transformer attention module that is being wrapped.
        """
        assert not isinstance(model, HRRAdaptedAttention)
        super().__init__()
        self.model_type = model_type
        self.model = model
        # Assume all parameters of the attention model we are wrapping are on the same device.
        device = next(model.parameters()).device
        # Don't think this was specified in the paper, but we follow the official repo which used an Embedding
        # which initializes the tokens with standard normal values.
        # https://github.com/ZrrSkywalker/LLaMA-Adapter/blob/41c3546fe1997ab8a65809dc8d8f9252b19d9faf/llama/model.py#L234
        # (bsz, adapter_len, hidden_size)
        target_dtype = torch.float32
        # target_dtype = (
        #     model.q_proj.weight.dtype if model.q_proj.weight.dtype not in [torch.int8, torch.uint8] else torch.float32
        # )

        # Initialize the gate to 0 as this is "zero-init".
        self.hrraa_adaption_gate = nn.Parameter(torch.zeros(1, device=device, dtype=target_dtype))

        # HRRAA
        self.hrraa_query = nn.Sequential(
            nn.Linear(self.model.hidden_size, self.model.hidden_size),
        )
        self.hrraa_key = nn.Sequential(
            nn.Linear(self.model.hidden_size, self.model.hidden_size),
        )
        self.hrraa_value = nn.Sequential(
            nn.Linear(self.model.hidden_size, self.model.hidden_size),
        )

    def forward(self, *args, **kwargs):
        """
        Forward pass for the adapter which wraps the original LlamaAttention module.

        Args:
            kwargs: See the original LlamaAttention module.
        """
        if kwargs.get("output_attention", False):
            raise NotImplementedError("output_attention is not currently supported.")

        output, *rest_original_output = self.model(*args, **kwargs)
        bsz = output.shape[0]
        q_len = output.shape[1]
        embed_dim = output.shape[2]

        # apply HRRAA
        get_inputs = TRANSFORMERS_MODEL_CONFIG[self.model_type].get_inputs
        hidden_states = get_inputs(*args, **kwargs)
        # hidden_states = kwargs.get("hidden_states")
        k = self.hrraa_key(hidden_states)
        v = self.hrraa_value(hidden_states)
        q = self.hrraa_query(hidden_states)
        values = hrr.key_value_query(k, v, q, causal=True)
        values = values.view(bsz, q_len, embed_dim)

        adapter_output = self.hrraa_adaption_gate * values
        output = output + adapter_output

        # Restore original dtype.
        # output = output.to(previous_dtype)
        return output, *rest_original_output

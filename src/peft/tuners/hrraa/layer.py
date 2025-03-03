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
    """
    This module adds a new, trainable attention layer that wraps an existing
    attention layer. The attention is based on neuro-symbolic systems and implemented in linear time wrt sequence length.
    """

    def __init__(self, model_type: str, model, is_causal=True):
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
        self.is_causal = is_causal
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
        Forward pass for the adapter which wraps the original attention module.

        Args:
            kwargs: See the original attention module.
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
        previous_dtype = hidden_states.dtype

        k = self.hrraa_key(hidden_states)
        v = self.hrraa_value(hidden_states)
        q = self.hrraa_query(hidden_states)

        k, q, v = map(lambda x: x.to(torch.float32), (k, q, v))

        adapter_output = self.attend(q, k, v)

        output = output + self.hrraa_adaption_gate * adapter_output
        # Restore original dtype.
        output = output.to(previous_dtype)
        return output, *rest_original_output

    def attend(self, q, k, v):
        bsz, q_len, embed_dim = v.shape
        values_hat = hrr.key_value_query(k, v, q, causal=self.is_causal, norm_kv=True)
        return values_hat


class HRRAdaptedAttentionRecastHRR(HRRAdaptedAttention):
    """
    HRRAA using the method from:
    Recasting Self-Attention with Holographic Reduced Representations
    https://arxiv.org/pdf/2305.19534.pdf
    """
    def attend(self, q, k, v):
        bsz, q_len, embed_dim  = v.shape
        values_hat = hrr.key_value_query(k, v, q, causal=True)
        # trying the softmax clean-up from https://arxiv.org/pdf/2305.19534.pdf
        values_presence = F.cosine_similarity(v, values_hat, -1)[..., None]
        values_weight = F.softmax(values_presence, -2)
        values = values_weight * v
        return values


HRRAA_ADAPTERS = dict(
    base=HRRAdaptedAttention,
    recast=HRRAdaptedAttentionRecastHRR,
)
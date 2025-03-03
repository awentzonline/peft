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

from collections import namedtuple
from dataclasses import dataclass, field

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class HRRAAConfig(PeftConfig):
    """Stores the configuration of an [`HRRAAModel`]."""

    target_modules: str = field(
        default=None, metadata={"help": "Name of the attention submodules to adapt."}
    )
    adapter_layers: int = field(default=1, metadata={"help": "Number of adapter layers (from the top)"})
    adapter_type: str = field(
        default='base', metadata={"help": "Variant of HRRAA (base, recast)"}
    )
    is_causal: bool = field(default=True, metadata={"help": "Is attention causal?"})
    def __post_init__(self):
        self.peft_type = PeftType.HRRAA

    @property
    def is_hrraa(self) -> bool:
        """Return True if this is a HRRAA config."""
        return True


# Contains the config that is specific to a transformers model type.
ModelTypeConfig = namedtuple(
    "ModelTypeConfig", ["target_modules", "get_inputs"]
)


def get_input_default(*args, **kwargs):
    return kwargs.get('hidden_states')


def get_input_bloom(*args, **kwargs):
    return args[0]


# Mapping of transformers model types to their specific configuration.
TRANSFORMERS_MODEL_CONFIG = {
    "bloom": ModelTypeConfig(
        target_modules="self_attention",
        get_inputs=get_input_bloom,
    ),
    "gpt2": ModelTypeConfig(
        target_modules="attn",
        get_inputs=get_input_default,
    ),
    "phi": ModelTypeConfig(
        target_modules="self_attn",
        get_inputs=get_input_default,
    ),
    "llama": ModelTypeConfig(
        target_modules="self_attn",
        get_inputs=get_input_default,
    ),
    "mistral": ModelTypeConfig(  # same as llama,
        target_modules="self_attn",
        get_inputs=get_input_default,
    ),
    "bert": ModelTypeConfig(
        target_modules="attention",
        get_inputs=get_input_default,
    ),
    "default": ModelTypeConfig(
        target_modules="self_attention",
        get_inputs=get_input_bloom,
    ),
}


def prepare_config(
    peft_config: HRRAAConfig,
    model,
) -> HRRAAConfig:
    """Prepare the config based on the llama model type."""
    if model.config.model_type not in TRANSFORMERS_MODEL_CONFIG:
        raise ValueError("Unsupported model type for adaption prompt: '{model.config.model_type}'.")

    model_config = TRANSFORMERS_MODEL_CONFIG[model.config.model_type]

    if peft_config.target_modules is None:
        peft_config.target_modules = model_config.target_modules

    return peft_config


def prepare_config(
    peft_config: HRRAAConfig,
    model,
) -> HRRAAConfig:
    return peft_config

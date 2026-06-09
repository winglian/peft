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


from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import (
    BaseTuner,
    get_device_map,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_OFT_TARGET_MODULES_MAPPING,
    get_quantization_config,
)

from .aqlm import dispatch_aqlm
from .awq import dispatch_awq
from .eetq import dispatch_eetq
from .gptq import dispatch_gptq
from .hqq import dispatch_hqq
from .inc import dispatch_inc
from .layer import OFTLayer, dispatch_default


class OFTModel(BaseTuner):
    """
    Creates Orthogonal Finetuning model from a pretrained model. The method is described in
    https://huggingface.co/papers/2306.07280

    Args:
        model (`torch.nn.Module`): The model to which the adapter tuner layers will be attached.
        config ([`OFTConfig`]): The configuration of the OFT model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The OFT model.

    Example:
        ```py
        >>> from diffusers import StableDiffusionPipeline
        >>> from peft import OFTModel, OFTConfig

        >>> config_te = OFTConfig(
        ...     r=8,
        ...     target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
        ...     module_dropout=0.0,
        ...     init_weights=True,
        ... )
        >>> config_unet = OFTConfig(
        ...     r=8,
        ...     target_modules=[
        ...         "proj_in",
        ...         "proj_out",
        ...         "to_k",
        ...         "to_q",
        ...         "to_v",
        ...         "to_out.0",
        ...         "ff.net.0.proj",
        ...         "ff.net.2",
        ...     ],
        ...     module_dropout=0.0,
        ...     init_weights=True,
        ... )

        >>> model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> model.text_encoder = OFTModel(model.text_encoder, config_te, "default")
        >>> model.unet = OFTModel(model.unet, config_unet, "default")
        ```

    **Attributes**:
        - **model** ([`~torch.nn.Module`]) -- The model to be adapted.
        - **peft_config** ([`OFTConfig`]): The configuration of the OFT model.
    """

    prefix: str = "oft_"
    tuner_layer_cls = OFTLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_OFT_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        oft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        kwargs = {
            "r": oft_config.r,
            "fan_in_fan_out": oft_config.fan_in_fan_out,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        quant_methods = ["gptq", "aqlm", "awq"]
        for quant_method in quant_methods:
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                kwargs[f"{quant_method}_quantization_config"] = quantization_config

        # If it is not a OFTLayer, create a new module, else update it with new adapters
        if not isinstance(target, OFTLayer):
            device_map = get_device_map(self.model)
            new_module = self._create_new_module(oft_config, adapter_name, target, device_map=device_map, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(
                adapter_name,
                r=oft_config.r,
                config=oft_config,
            )

    def _create_and_replace_param(self, oft_config, module, parameter_name, adapter_name):
        # Input-side OFT for a fused MoE expert nn.Parameter: parametrize it so it
        # presents as a RotatedGroupedWeight (rotates activations, base stays packed).
        from torch.nn.utils import parametrize

        from .experts import OFTGroupedParametrization

        weight = getattr(module, parameter_name)
        if weight.dim() != 3:
            raise ValueError(
                f"OFT `target_parameters` expects a 3-D fused expert weight (num_experts, out, in), "
                f"but '{parameter_name}' has shape {tuple(weight.shape)}."
            )
        num_experts, _, in_features = weight.shape
        split = 2 if "gate_up" in parameter_name else 1
        param = OFTGroupedParametrization(num_experts, in_features, split, oft_config, adapter_name).to(
            device=weight.device, dtype=weight.dtype
        )
        parametrize.register_parametrization(module, parameter_name, param, unsafe=True)

    def _inject_parameters(self, peft_config, model, adapter_name, low_cpu_mem_usage):
        # Snapshot matches before injecting: registering a parametrization mutates the
        # module's `_parameters`, which would break iteration over `named_parameters`.
        target_names = sorted(set(peft_config.target_parameters))
        matches = []
        for module_name, module in model.named_modules():
            for param_name, _ in list(module.named_parameters(recurse=False)):
                key = f"{module_name}.{param_name}"
                if key in target_names or any(key.endswith(f".{name}") for name in target_names):
                    matches.append((module, param_name, key))
        for module, param_name, key in matches:
            self._create_and_replace_param(peft_config, module, param_name, adapter_name)
            self.targeted_parameter_names.append(key)

    def _prepare_adapter_config(self, peft_config, model_config):
        if peft_config.target_modules is None:
            target_modules = self.target_module_mapping.get(model_config["model_type"])
            if target_modules is not None:
                peft_config.target_modules = target_modules if isinstance(target_modules, str) else set(target_modules)
            elif not peft_config.target_parameters:
                raise ValueError("Please specify `target_modules` or `target_parameters` in `peft_config`")
        return peft_config

    @staticmethod
    def _create_new_module(oft_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced OFT layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = []

        # avoid eager bnb import
        if is_bnb_available():
            from .bnb import dispatch_bnb_8bit

            dispatchers.append(dispatch_bnb_8bit)

        if is_bnb_4bit_available():
            from .bnb import dispatch_bnb_4bit

            dispatchers.append(dispatch_bnb_4bit)

        dispatchers.extend(
            [
                dispatch_eetq,
                dispatch_aqlm,
                dispatch_awq,
                dispatch_gptq,
                dispatch_hqq,
                dispatch_inc,
                dispatch_default,
            ]
        )

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, oft_config=oft_config, **kwargs)
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Conv2d`."
            )

        return new_module

    def _check_merge_allowed(self):
        """Verify that the configuration supports merging.

        Currently gptq quantization and replicated layers do not support merging.
        """
        super()._check_merge_allowed()
        if getattr(self.model, "quantization_method", None) == "gptq":
            raise ValueError("Cannot merge OFT layers when the model is gptq quantized")
        if self.peft_config.get("layer_replication"):
            raise ValueError("Cannot merge OFT layers when base model layers are replicated")

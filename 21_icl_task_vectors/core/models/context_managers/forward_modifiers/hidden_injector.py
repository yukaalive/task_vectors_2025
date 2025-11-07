import torch
from transformers import PreTrainedModel

from core.models.utils.llm_layers import get_layers


class HiddenInjector:
    def __init__(
        self,
        model: PreTrainedModel,
        injection_layers: torch.Tensor,  # (batch_size)
        injection_positions: torch.Tensor,  # (batch_size)
        hiddens_to_inject: torch.Tensor,  # (batch_size, hidden_size)
    ):
        """
        Args:
            model: The model to inject hidden states into
            injection_layer: the layer to inject hidden states into, for each example in the batch (batch_size)
            injection_position: the position to inject hidden states into, for each example in the batch (batch_size)
            hidden_to_inject: the hidden states to inject, for each example in the batch (batch_size, hidden_size)
        """

        self._model = model
        self._injection_layer = injection_layers
        self._injection_position = injection_positions
        self._hidden_to_inject = hiddens_to_inject

        self._hooks = []

    def __enter__(self):
        self._register_forward_hooks()

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self._hooks:
            hook.remove()

    def _register_forward_hooks(self):
        def inject_hidden_hook(layer_idx):
            def inject_hidden(mod, inp, out):
                hidden_states = out[0] if isinstance(out, tuple) else out
                # このレイヤーで注入すべきサンプルを特定（特定のレイヤーだけをtrueに）
                mask = self._injection_layer == layer_idx
                # print("maskのlen：",mask)

                if mask.any():
                    # 注入する値のデバイス（CPU/GPU）とデータ型（float32/float16等）をhidden_statesに合わせる
                    hidden_to_inject = self._hidden_to_inject.to(hidden_states.device).type(hidden_states.dtype)
                    # 注入対象のサンプル番号を取得
                    idx_to_inject = torch.arange(hidden_states.shape[0], device=hidden_states.device)[mask]
                     # 実際の注入: 指定したサンプルの指定した位置に値を上書き
                    # hidden_states[サンプル番号, 位置] = 注入値
                    # 左辺: idx_to_injectで指定されたサンプルの、_injection_position[mask]で指定された位置
                    # 右辺: hidden_to_inject[mask]で取り出した注入値
                    hidden_states[idx_to_inject, self._injection_position[mask]] = hidden_to_inject[mask]

                return out

            return inject_hidden

        for i, layer in enumerate(get_layers(self._model)):
            hook = layer.register_forward_hook(inject_hidden_hook(i))
            self._hooks.append(hook)

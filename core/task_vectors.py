from typing import Dict, List, Optional, Tuple, Union, Iterable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from core.analysis.evaluation import calculate_accuracy_on_datasets
from core.data.datasets.few_shot_dataset import FewShotDataset
from core.data.tasks.task import Task
from core.models.context_managers.forward_modifiers.hidden_injector import HiddenInjector
from core.models.utils.inference import (
    batch_forward,
    batch_generate,
    decode_predictions,
    get_input_type,
    modified_forward,
    tokenize_datasets,
    traced_forward,
)
from core.models.utils.llm_layers import get_layers
from core.utils.nested import nested_apply


def run_icl(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    test_datasets: List[FewShotDataset],
    include_train: bool = True,
    generation_mode: str = "multi",
) -> List[str]:
    format_dataset_kwargs = {"include_train": include_train}
    inputs = tokenize_datasets(tokenizer, test_datasets, format_dataset_kwargs=format_dataset_kwargs)
    max_new_tokens = 1 if generation_mode == "single" else 30
    new_ids = batch_generate(model, tokenizer, inputs=inputs, generate_kwargs={"max_new_tokens": max_new_tokens})
    predictions = decode_predictions(new_ids, tokenizer)
    return predictions


def run_task_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    test_datasets: List[FewShotDataset],
    dev_datasets: List[FewShotDataset],
    layers_to_test: Optional[Iterable[int]] = None,
    multi_context: bool = False,
    generation_mode: str = "multi",
    max_new_tokens: int = 30,
):
    dev_accuracy_by_layer = task_vector_accuracy_by_layer(
        model,
        tokenizer,
        task,
        dev_datasets,
        layers_to_test=layers_to_test,
        multi_context=multi_context,
        generation_mode=generation_mode,
        max_new_tokens=max_new_tokens,
    )
    best_intermediate_layer = int(max(dev_accuracy_by_layer, key=dev_accuracy_by_layer.get))
    print("ベストレイヤー：",best_intermediate_layer)
    task_hiddens = get_task_hiddens(model, tokenizer, task, test_datasets, multi_context=multi_context)
    predictions = modulated_generate(
        model,
        tokenizer,
        task,
        test_datasets,
        task_hiddens=task_hiddens,
        intermediate_layer=best_intermediate_layer,
        generation_mode=generation_mode,
        max_new_tokens=max_new_tokens,
    )

    return predictions, dev_accuracy_by_layer, task_hiddens


# def get_multi_context_task_hiddens(
#     model: PreTrainedModel,
#     tokenizer: PreTrainedTokenizer,
#     task: Task,
#     datasets: List[FewShotDataset],
# ) -> torch.Tensor:
#     inputs = tokenize_datasets(tokenizer, datasets)

#     outputs, forward_trace = traced_forward(model, inputs=inputs)

#     task_hiddens = forward_trace.residual_stream.hidden[:, :, -1, :]

#     # for each dataset, average task hiddens from other datasets that did not include the test_input from the current dataset
#     mask = torch.ones(len(datasets), len(datasets))
#     for i, dataset in enumerate(datasets):
#         for j, other_dataset in enumerate(datasets):
#             if dataset.test_input in other_dataset.train_inputs or dataset.test_input == other_dataset.test_input:
#                 mask[i, j] = 0

#     task_hiddens = torch.cat([task_hiddens[mask[i].bool()].mean(dim=0).unsqueeze(0) for i in range(len(datasets))])

#     task_hiddens = task_hiddens[:, 1:]  # the first one is the embedding layer
#     return task_hiddens  # (num_datasets, num_layers, hidden_size)


def get_single_context_task_hiddens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
    num_test_inputs_to_avg: int = 2,
) -> torch.Tensor:
    # print("★★★★★★★★★★★★★★★★★★★★★★★★★★★★★")
    # print("datasetsのサイズ：",len(datasets))
    d = datasets[0]
    # print("FewShotDatasetのtrain_inputs:", d.train_inputs)
    # print("FewShotDatasetのtest_input:", d.test_input)
    new_datasets = []
    for dataset in datasets:
        test_inputs_list = list(task.sample_inputs(num_test_inputs_to_avg, exclude=(dataset.test_input,)))
        for test_input in test_inputs_list:
            # 新しいFewShotDatasetを作成
            new_dataset = FewShotDataset(
                train_inputs=dataset.train_inputs,
                train_outputs=dataset.train_outputs,
                test_input=test_input,
                test_output=task.calc_output(test_input),
            )
            new_datasets.append(new_dataset)
    # print("new_datasetsのshape2：",len(new_datasets))
    e0 = new_datasets[0]
    e1 = new_datasets[1]
    # print("new_datasetsのtrain_inputs:", e0.train_inputs)
    # print("new_datasetsのtest_input:", e0.test_input)
    # print("new_datasetsのtrain_inputs:", e1.train_inputs)
    # print("new_datasetsのtest_input:", e1.test_input)
    inputs = tokenize_datasets(tokenizer, new_datasets)

    # TODO: replace traced forward with a regular forward and rely on huggingface's saved hidden states
    outputs, forward_trace = traced_forward(model, inputs=inputs)
    # print("全体ベクトル（プロンプト＋テスト）:", forward_trace.residual_stream.hidden.shape) 
    task_hiddens = forward_trace.residual_stream.hidden[:, :, -1, :]
    # print("抽出したタスクベクトル:", task_hiddens.shape) 
    task_hiddens_pre = forward_trace.residual_stream.hidden[:, :, -1, :]  
    # print("=== 平均前 (test_input ブレあり) ===")
    # print("shape:", task_hiddens_pre.shape)
    # print("layer0 の最初の10次元:", task_hiddens_pre[0, 0, :10])  # dataset0, layer0 例


    _, num_layers, hidden_size = task_hiddens.shape
    task_hiddens = task_hiddens.view(len(datasets), num_test_inputs_to_avg, num_layers, hidden_size).mean(dim=1)


    _, num_layers, hidden_size = task_hiddens_pre.shape
    task_hiddens_post = task_hiddens_pre.view(
        len(datasets), num_test_inputs_to_avg, num_layers, hidden_size
    ).mean(dim=1)

    # print("\n=== 平均後 (test_inputブレ除去・安定化後) ===")
    # print("shape:", task_hiddens_post.shape)
    # print("layer0 の最初の10次元:", task_hiddens_post[0, 0, :10])


    task_hiddens = task_hiddens[:, 1:]  # the first one is the embedding layer
    return task_hiddens  # (num_datasets, num_layers, hidden_size)




def get_task_hiddens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
    multi_context: bool = False,
) -> torch.Tensor:
    if multi_context:
        return get_multi_context_task_hiddens(model, tokenizer, task, datasets)
    else:
        return get_single_context_task_hiddens(model, tokenizer, task, datasets)


# def modulated_generate(
#     model: PreTrainedModel,
#     tokenizer: PreTrainedTokenizer,
#     task: Task,
#     test_datasets: List[FewShotDataset],
#     task_hiddens: torch.tensor,
#     intermediate_layer: Union[int, torch.Tensor],
#     past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
#     return_task_hiddens: bool = False,
#     include_train: bool = False,
#     max_new_tokens: int = 30, #//★変更★//
# ) -> List[str]:
#     inputs = tokenize_datasets(tokenizer, test_datasets, format_dataset_kwargs={"include_train": include_train})

#     first_forward_outputs = modulated_forward(
#         model,
#         inputs=inputs,
#         task_hiddens=task_hiddens,
#         intermediate_layer=intermediate_layer,
#         past_key_values=past_key_values,
#     )
#     first_predicted_token_ids = first_forward_outputs.logits[:, -1].argmax(dim=-1).unsqueeze(-1)
#     answers = decode_predictions(first_predicted_token_ids, tokenizer)

#     if return_task_hiddens:
#         return answers, task_hiddens
#     return answers
def modulated_generate_single(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    test_datasets: List[FewShotDataset],
    task_hiddens: torch.tensor,
    intermediate_layer: Union[int, torch.Tensor],
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    return_task_hiddens: bool = False,
    include_train: bool = False,
) -> List[str]:
    """Single token generation version (original implementation)"""
    inputs = tokenize_datasets(tokenizer, test_datasets, format_dataset_kwargs={"include_train": include_train})

    first_forward_outputs = modulated_forward(
        model,
        inputs=inputs,
        task_hiddens=task_hiddens,
        intermediate_layer=intermediate_layer,
        past_key_values=past_key_values,
    )
    first_predicted_token_ids = first_forward_outputs.logits[:, -1].argmax(dim=-1).unsqueeze(-1)
    answers = decode_predictions(first_predicted_token_ids, tokenizer)

    if return_task_hiddens:
        return answers, task_hiddens
    return answers


def modulated_generate_multi(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    test_datasets: List[FewShotDataset],
    task_hiddens: torch.tensor,
    intermediate_layer: Union[int, torch.Tensor],
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    return_task_hiddens: bool = False,
    include_train: bool = False,
    max_new_tokens: int = 30,
) -> List[str]:
    device = model.device
    inputs = tokenize_datasets(tokenizer, test_datasets, format_dataset_kwargs={"include_train": include_train})
    
    # 入力をデバイスに移動
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # ステップ1: task vectorを注入して最初の1トークンを生成
    first_forward_outputs = modulated_forward(
        model,
        inputs=inputs,
        task_hiddens=task_hiddens,
        intermediate_layer=intermediate_layer,
        past_key_values=past_key_values,
    )

    # Step 1 後（modulated_forward の直後）
    # print("first_forward_outputs.past_key_valuesの長さ",len(first_forward_outputs.past_key_values))  # = num_layers
    # print("first_forward_outputs.past_key_values[0][0]のshape]",first_forward_outputs.past_key_values[0][0].shape)  # [batch, num_heads, seq_len, head_dim]
    # print("first_forward_outputs.past_key_values[0][1]のshape]",first_forward_outputs.past_key_values[0][1].shape)


    first_predicted_token_ids = first_forward_outputs.logits[:, -1].argmax(dim=-1).unsqueeze(-1)
    first_predicted_token_ids = first_predicted_token_ids.to(device)
    # max_new_tokens=1の場合は1トークンのみ返す
    if max_new_tokens <= 1:
        answers = decode_predictions(first_predicted_token_ids, tokenizer)
        if return_task_hiddens:
            return answers, task_hiddens
        return answers

    # ステップ2: 残りのトークンを生成（max_new_tokens > 1の場合）
    new_input_ids = first_predicted_token_ids
    new_attention_mask = torch.ones_like(new_input_ids).to(device)
    # 新しく生成されたトークンを既存の入力の末尾にくっつけたinput
    full_input_ids = torch.cat([inputs["input_ids"], new_input_ids], dim=-1)
    full_attention_mask = torch.cat([inputs["attention_mask"], new_attention_mask], dim=-1)

    # past_key_valuesをデバイスに移動
    updated_past_key_values = first_forward_outputs.past_key_values

    # DEBUG: Check past_key_values
    has_past_kv = updated_past_key_values is not None
    # print(f"DEBUG modulated_generate_multi: has past_key_values={has_past_kv}")
    if not has_past_kv:
        # If no past_key_values, we need to do full forward without cache
        # print(f"DEBUG: No past_key_values, using full input for generation")
        output_ids = model.generate(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens - 1,  # 既に1トークン生成済み
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        if updated_past_key_values is not None:
            updated_past_key_values = nested_apply(
                updated_past_key_values,
                lambda x: x.to(device) if isinstance(x, torch.Tensor) else x
            )

        # 残りのトークンを生成（max_new_tokens - 1トークン）
        output_ids = model.generate(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens - 1,  # 既に1トークン生成済み
            past_key_values=updated_past_key_values,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 元の入力部分を除いて新しく生成された部分のみを取得
    new_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
    answers = decode_predictions(new_ids, tokenizer)
    if return_task_hiddens:
        return answers, task_hiddens
    return answers


def modulated_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    test_datasets: List[FewShotDataset],
    task_hiddens: torch.tensor,
    intermediate_layer: Union[int, torch.Tensor],
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    return_task_hiddens: bool = False,
    include_train: bool = False,
    generation_mode: str = "multi",
    max_new_tokens: int = 30,
) -> List[str]:
    """
    Unified interface for modulated generation.

    Args:
        generation_mode: "single" for 1-token generation, "multi" for multi-token generation
    """
    if generation_mode == "single":
        return modulated_generate_single(
            model=model,
            tokenizer=tokenizer,
            task=task,
            test_datasets=test_datasets,
            task_hiddens=task_hiddens,
            intermediate_layer=intermediate_layer,
            past_key_values=past_key_values,
            return_task_hiddens=return_task_hiddens,
            include_train=include_train,
        )
    elif generation_mode == "multi":
        return modulated_generate_multi(
            model=model,
            tokenizer=tokenizer,
            task=task,
            test_datasets=test_datasets,
            task_hiddens=task_hiddens,
            intermediate_layer=intermediate_layer,
            past_key_values=past_key_values,
            return_task_hiddens=return_task_hiddens,
            include_train=include_train,
            max_new_tokens=max_new_tokens,
        )
    else:
        raise ValueError(f"Invalid generation_mode: {generation_mode}. Must be 'single' or 'multi'")


def modulated_forward(
    model: PreTrainedModel,
    inputs: Dict,
    task_hiddens: torch.Tensor,
    intermediate_layer: int,
    batch_size: Optional[int] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
):
    
    device = model.device
    """"""
    # 入力とタスク隠れ層をデバイスに移動
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    task_hiddens = task_hiddens.to(device)

    # past_key_valuesをデバイスに移動（ここを追加）
    if past_key_values is not None:
        past_key_values = nested_apply(
            past_key_values, 
            lambda x: x.to(device) if isinstance(x, torch.Tensor) else x
        )
    """"""
    # TODO: move all this to the HiddenInjector class
    # intermediate_layerが整数の場合、バッチサイズ分だけ複製
    # 例: 1 → [1, 1] (バッチサイズ2の場合)
    if isinstance(intermediate_layer, int):
        intermediate_layer = torch.tensor(intermediate_layer).repeat(len(inputs["input_ids"]))
        # print("intermediate_layerのshape",intermediate_layer.shape)

    # intermediate_layerと同じshapeで、全て-1のテンソルを作成
    # -1は「文章の最後の位置」を意味する
    # 例: intermediate_layer=[1,1] → injection_positions=[-1,-1]
    injection_positions = -1 * torch.ones_like(intermediate_layer, dtype=torch.long)

    task_hiddens = task_hiddens[torch.arange(len(intermediate_layer)), intermediate_layer]
    # print("task_hiddensのshape",task_hiddens.shape)
    forward_modifiers = [
        HiddenInjector(
            model,
            injection_layers=intermediate_layer,# どの層に注入するか [5, 5]
            injection_positions=injection_positions,# どの位置に注入するか [-1, -1]
            hiddens_to_inject=task_hiddens,# 何を注入するか (2, 4096)
        )
    ]

    if past_key_values is not None:
        inputs[get_input_type(inputs)] = inputs[get_input_type(inputs)][:, -1].unsqueeze(1)

    first_forward_outputs = modified_forward(
        model,
        inputs=inputs,
        forward_kwargs={"past_key_values": past_key_values, "use_cache": True},  # Add use_cache=True
        forward_modifiers=forward_modifiers,
        batch_size=len(inputs["input_ids"]),  # TODO: need to enable batched forward with HiddenInjector
    )

    return first_forward_outputs


def task_vector_accuracy_by_layer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
    layers_to_test: Optional[Iterable[int]] = None,
    multi_context: bool = False,
    generation_mode: str = "multi",
    max_new_tokens: int = 30,
) -> Dict[int, float]:
    if layers_to_test is None:
        num_layers = len(get_layers(model))
        layers_to_test = range(num_layers)

    # Get task hiddens
    task_hiddens = get_task_hiddens(model, tokenizer, task, datasets, multi_context=multi_context)
    # GPUに移動
    device = model.device
    task_hiddens = task_hiddens.to(device)
    # Get input past_key_values
    inputs = tokenize_datasets(tokenizer, datasets, format_dataset_kwargs={"include_train": False})
    outputs = batch_forward(model, inputs=inputs, forward_kwargs={"use_cache": True})
    past_key_values = outputs.past_key_values
    past_key_values = nested_apply(past_key_values, lambda x: x[:, :, :-1])  # remove last token from past_key_values
    inputs["input_ids"] = inputs["input_ids"][:, -1].unsqueeze(1)

    # Find best intermediate layer using dev set
    accuracies = []
    for layer_num in layers_to_test:
        answers = modulated_generate(
            model,
            tokenizer,
            task,
            datasets,
            intermediate_layer=layer_num,
            task_hiddens=task_hiddens,
            past_key_values=past_key_values,
            generation_mode=generation_mode,
            max_new_tokens=max_new_tokens,
        )

        accuracy = calculate_accuracy_on_datasets(task, answers, datasets)
        accuracies.append(accuracy)
    accuracy_by_layer = {layer: accuracy for layer, accuracy in zip(layers_to_test, accuracies)}

    return accuracy_by_layer


def run_cross_task_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    source_task: Task,
    target_task: Task,
    source_dev_datasets: List[FewShotDataset],
    target_test_datasets: List[FewShotDataset],
    target_dev_datasets: List[FewShotDataset],
    layers_to_test: Optional[Iterable[int]] = None,
    multi_context: bool = False,
    generation_mode_source: str = "single",
    generation_mode_target: str = "multi",
    max_new_tokens_source: int = 1,
    max_new_tokens_target: int = 30,
):
    """
    Cross-task vector transfer experiment.

    1. Find best layer for source task using source dev datasets
    2. Extract task vector from source task at its best layer
    3. Apply source task vector to target task across all layers
    4. Find best target layer using target dev datasets
    5. Generate predictions on target test datasets

    Args:
        model: The language model
        tokenizer: The tokenizer
        source_task: Source task (e.g., ja_en_single)
        target_task: Target task (e.g., ja_en_easy)
        source_dev_datasets: Development datasets for source task
        target_test_datasets: Test datasets for target task
        target_dev_datasets: Development datasets for target task
        layers_to_test: Layers to test (default: all layers)
        multi_context: Whether to use multi-context
        generation_mode_source: Generation mode for source task
        generation_mode_target: Generation mode for target task
        max_new_tokens_source: Max new tokens for source task
        max_new_tokens_target: Max new tokens for target task

    Returns:
        predictions: Predictions on target test datasets
        source_best_layer: Best layer for source task
        source_dev_accuracy_by_layer: Source task dev accuracy by layer
        target_dev_accuracy_by_layer: Target task dev accuracy by layer (with source vector)
        source_task_hiddens: Task hiddens from source task
    """
    # Step 1: Find best layer for source task
    print("Step 1: Finding best layer for source task...")
    source_dev_accuracy_by_layer = task_vector_accuracy_by_layer(
        model,
        tokenizer,
        source_task,
        source_dev_datasets,
        layers_to_test=layers_to_test,
        multi_context=multi_context,
        generation_mode=generation_mode_source,
        max_new_tokens=max_new_tokens_source,
    )
    source_best_layer = int(max(source_dev_accuracy_by_layer, key=source_dev_accuracy_by_layer.get))
    print(f"Source task best layer: {source_best_layer}")

    # Step 2: Extract task vectors from source dev datasets
    print("Step 2: Extracting task vectors from source dev datasets...")
    # Use source task with source dev datasets (each has 5 demos)
    source_task_hiddens = get_task_hiddens(
        model, tokenizer, source_task, source_dev_datasets, multi_context=multi_context
    )
    device = model.device
    source_task_hiddens = source_task_hiddens.to(device)
    print(f"Source task hiddens shape: {source_task_hiddens.shape}")
    print(f"  → {len(source_dev_datasets)} task vectors extracted from source dev datasets")

    # Step 3: Get past_key_values for target dev datasets (same as existing code)
    print("Step 3: Computing past_key_values for target dev datasets...")
    target_dev_inputs = tokenize_datasets(tokenizer, target_dev_datasets, format_dataset_kwargs={"include_train": False})
    target_dev_outputs = batch_forward(model, inputs=target_dev_inputs, forward_kwargs={"use_cache": True})
    target_dev_past_key_values = target_dev_outputs.past_key_values
    target_dev_past_key_values = nested_apply(target_dev_past_key_values, lambda x: x[:, :, :-1])  # remove last token
    target_dev_inputs["input_ids"] = target_dev_inputs["input_ids"][:, -1].unsqueeze(1)

    # Step 4: Find best target layer using SOURCE task vectors on target dev set
    print("Step 4: Finding best target layer using SOURCE task vectors on target dev...")
    # Use source_task_hiddens to find the best layer for target task
    # This is the cross-task transfer approach

    if layers_to_test is None:
        num_layers = len(get_layers(model))
        layers_to_test = range(num_layers)

    # Test each target layer with SOURCE task vectors
    print(f"Testing SOURCE task vectors on target dev across {len(list(layers_to_test))} layers...")
    target_dev_accuracy_by_layer = {}
    for target_layer_num in layers_to_test:
        print(f"  Testing layer {target_layer_num}...")
        answers = modulated_generate(
            model,
            tokenizer,
            target_task,
            target_dev_datasets,
            intermediate_layer=target_layer_num,
            task_hiddens=source_task_hiddens,  # Use SOURCE task vectors for cross-task transfer
            past_key_values=target_dev_past_key_values,  # Use target task's past_key_values
            generation_mode=generation_mode_target,
            max_new_tokens=max_new_tokens_target,
        )

        accuracy = calculate_accuracy_on_datasets(target_task, answers, target_dev_datasets)
        target_dev_accuracy_by_layer[target_layer_num] = accuracy
        print(f"    → Accuracy: {accuracy:.2f}")

    # Find target best layer
    max_accuracy = max(target_dev_accuracy_by_layer.values())
    if max_accuracy == 0.0:
        # If all layers have 0.0 accuracy, use source best layer as fallback
        target_best_layer = source_best_layer
        print(f"All target layers have 0.00 accuracy, using source best layer: {target_best_layer}")
    else:
        target_best_layer = int(max(target_dev_accuracy_by_layer, key=target_dev_accuracy_by_layer.get))
        print(f"Target task best layer: {target_best_layer}")
        print(f"  Max accuracy (with SOURCE task vectors): {target_dev_accuracy_by_layer[target_best_layer]:.2f}")

    # Step 5: Get past_key_values for target test datasets
    print("Step 5: Computing past_key_values for target test datasets...")
    target_test_inputs = tokenize_datasets(tokenizer, target_test_datasets, format_dataset_kwargs={"include_train": False})
    target_test_outputs = batch_forward(model, inputs=target_test_inputs, forward_kwargs={"use_cache": True})
    target_test_past_key_values = target_test_outputs.past_key_values
    target_test_past_key_values = nested_apply(target_test_past_key_values, lambda x: x[:, :, :-1])  # remove last token
    target_test_inputs["input_ids"] = target_test_inputs["input_ids"][:, -1].unsqueeze(1)

    # Step 6: Generate predictions on target test datasets with SOURCE task vectors
    print("Step 6: Generating predictions with SOURCE task vectors on target test datasets...")
    print(f"  Using SOURCE task vectors at target best layer {target_best_layer}")
    predictions = modulated_generate(
        model,
        tokenizer,
        target_task,
        target_test_datasets,
        task_hiddens=source_task_hiddens,  # Use SOURCE task vectors for cross-task transfer
        past_key_values=target_test_past_key_values,  # Use target task's past_key_values
        intermediate_layer=target_best_layer,
        generation_mode=generation_mode_target,
        max_new_tokens=max_new_tokens_target,
    )

    return (
        predictions,
        source_best_layer,
        target_best_layer,
        source_dev_accuracy_by_layer,
        target_dev_accuracy_by_layer,
        source_task_hiddens,
    )


import json
from typing import Any
import torch

def json2token(
    obj: Any,
    sort_json_key: bool = False,
    all_special_tokens: list[str] | None = None,
):
    tokens_to_add = set()

    if all_special_tokens is None:
        all_special_tokens = []

    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"], tokens_to_add

        output = ""

        if sort_json_key:
            keys = sorted(obj.keys(), reverse=True)
        else:
            keys = obj.keys()

        for k in keys:
            tokens_to_add.update([
                f"<s_{k}>",
                f"</s_{k}>",
            ])

            value, child_tokens = json2token(
                obj[k],
                sort_json_key,
                all_special_tokens,
            )

            tokens_to_add.update(child_tokens)

            output += (
                f"<s_{k}>"
                + value
                + f"</s_{k}>"
            )

        return output, tokens_to_add

    elif type(obj) == list:
        outputs = []

        for item in obj:
            value, child_tokens = json2token(
                item,
                sort_json_key,
                all_special_tokens,
            )

            outputs.append(value)
            tokens_to_add.update(child_tokens)

        return r"<sep/>".join(outputs), tokens_to_add

    else:
        obj = str(obj)

        if f"<{obj}/>" in all_special_tokens:
            obj = f"<{obj}/>"

        return obj, tokens_to_add
    
def process_ground_truth(tokenizer, ground_truth, sort_json_key = False, task_start_token="<s>"):
    ground_truth = json.loads(ground_truth)
    output, list_of_tokens = json2token(
        ground_truth,
        sort_json_key=sort_json_key,
        all_special_tokens=tokenizer.all_special_tokens
    )

    gt_token_sequence = task_start_token + output + tokenizer.eos_token
    
    added_tokens = set()
    added_tokens.update(list_of_tokens)
    return gt_token_sequence, sorted(added_tokens)

def get_train_inputs(tokenizer, processed_parse, prompt_end_token_id, max_length = 768, ignore_id = -100):
    input_ids = tokenizer(
        processed_parse,
        add_special_tokens=False,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"].squeeze(0)
    
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = ignore_id 
    labels[: torch.nonzero(labels == prompt_end_token_id).sum() + 1] = ignore_id    
    return input_ids, labels
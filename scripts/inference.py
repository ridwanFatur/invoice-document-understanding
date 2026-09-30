from torch.nn.utils.rnn import pad_sequence
from transformers.file_utils import ModelOutput
import re

def token2json(tokens, is_inner_value=False):
    output = dict()

    while tokens:
        start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
        if start_token is None:
            break
        key = start_token.group(1)
        end_token = re.search(fr"</s_{key}>", tokens, re.IGNORECASE)
        start_token = start_token.group()
        if end_token is None:
            tokens = tokens.replace(start_token, "")
        else:
            end_token = end_token.group()
            start_token_escaped = re.escape(start_token)
            end_token_escaped = re.escape(end_token)
            content = re.search(f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE)
            if content is not None:
                content = content.group(1).strip()
                if r"<s_" in content and r"</s_" in content:  # non-leaf node
                    value = token2json(content, is_inner_value=True)
                    if value:
                        if len(value) == 1:
                            value = value[0]
                        output[key] = value
                else:  # leaf nodes
                    output[key] = []
                    for leaf in content.split(r"<sep/>"):
                        leaf = leaf.strip()
                        if (
                            leaf in tokenizer.get_added_vocab()
                            and leaf[0] == "<"
                            and leaf[-2:] == "/>"
                        ):
                            leaf = leaf[1:-2]  # for categorical special tokens
                        output[key].append(leaf)
                    if len(output[key]) == 1:
                        output[key] = output[key][0]

            tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
            if tokens[:6] == r"<sep/>":  # non-leaf nodes
                return [output] + token2json(tokens[6:], is_inner_value=True)

    if len(output):
        return [output] if is_inner_value else output
    else:
        return [] if is_inner_value else {"text_sequence": tokens}

def inference(model, tokenizer, image_tensors, prompt_tensors):
    last_hidden_state = model.encoder(image_tensors)
    encoder_outputs = ModelOutput(last_hidden_state=last_hidden_state.flatten(1, 2), attentions=None)
    decoder_output = donut_model.decoder.model.generate(
        decoder_input_ids=prompt_tensors,
        encoder_outputs=encoder_outputs,
        max_length=max_length,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        output_attentions=False,
    )  
    output = {"predictions": list()}
    for seq in tokenizer.batch_decode(decoder_output.sequences):
        seq = seq.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        output["predictions"].append(token2json(seq))
    return output

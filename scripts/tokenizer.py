from transformers import XLMRobertaTokenizer

def get_default_tokenizer():
    tokenizer = XLMRobertaTokenizer.from_pretrained(
        "hyunwoongko/asian-bart-ecjk"
    )
    list_of_tokens = ["<sep/>"]
    tokenizer.add_special_tokens({
        "additional_special_tokens": sorted(set(list_of_tokens))
    })    
    return tokenizer

def get_tokenizer(path = "./tokenizer"):
    tokenizer = XLMRobertaTokenizer.from_pretrained(path)
    return tokenizer
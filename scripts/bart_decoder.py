from typing import Any, List, Optional, Union
from transformers import MBartConfig, MBartForCausalLM
from transformers.file_utils import ModelOutput
import torch.nn as nn
import torch

class BARTDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer: int, 
        max_position_embeddings: int, 
        tokenizer,
    ):    
        super().__init__()
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = max_position_embeddings     
        self.tokenizer = tokenizer

        # Model
        self.model = MBartForCausalLM(
            config=MBartConfig(
                is_decoder=True,
                is_encoder_decoder=False,
                add_cross_attention=True,
                decoder_layers=self.decoder_layer,
                max_position_embeddings=self.max_position_embeddings,
                vocab_size=len(self.tokenizer),
                scale_embedding=True,
                add_final_layer_norm=True,
            )
        ) 
        self.model.forward = self.forward
        self.model.config.is_encoder_decoder = True
        self.model.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id
        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_inference
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def prepare_inputs_for_inference(
        self,
        input_ids: torch.Tensor,
        encoder_outputs=None,          
        past_key_values=None,
        past=None,                     
        use_cache: bool = None,
        attention_mask: torch.Tensor = None,
        **kwargs,                      
    ):
        # compatibility with older transformers
        if past is not None:
            past_key_values = past
    
        if attention_mask is None:
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
    
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
    
        if hasattr(encoder_outputs, "last_hidden_state"):
            encoder_hidden_states = encoder_outputs.last_hidden_state
        else:
            encoder_hidden_states = encoder_outputs
    
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "encoder_hidden_states": encoder_hidden_states,
        }
        
    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict
        outputs = self.model.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.model.lm_head(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ModelOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            decoder_attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
        
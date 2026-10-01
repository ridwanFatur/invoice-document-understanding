# donut.py
from typing import List, Optional
import torch
import torch.nn as nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    MBartConfig,
    MBartForCausalLM,
)
from transformers.modeling_outputs import ModelOutput
from timm.models.swin_transformer import SwinTransformer


# ============================================================
# 1. Swin Encoder
# ============================================================
class SwinEncoder(nn.Module):
    def __init__(
        self,
        input_size: List[int],
        window_size: int,
        encoder_layer: List[int],
    ):
        super().__init__()
        self.input_size = input_size
        self.window_size = window_size
        self.encoder_layer = encoder_layer

        self.model = SwinTransformer(
            img_size=self.input_size,
            depths=self.encoder_layer,
            window_size=self.window_size,
            patch_size=4,
            embed_dim=128,
            num_heads=[4, 8, 16, 32],
            num_classes=0,
        )
        self.model.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.patch_embed(x)
        x = self.model.layers(x)
        return x


# ============================================================
# 2. BART Decoder
# ============================================================
class BARTDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: int,
        max_position_embeddings: int,
        vocab_size: int,
        pad_token_id: int,
    ):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        self.model = MBartForCausalLM(
            config=MBartConfig(
                is_decoder=True,
                is_encoder_decoder=False,
                add_cross_attention=True,
                decoder_layers=self.decoder_layer,
                max_position_embeddings=self.max_position_embeddings,
                vocab_size=vocab_size,
                scale_embedding=True,
                add_final_layer_norm=True,
                pad_token_id=pad_token_id,
            )
        )

        self.model.forward = self.forward
        self.model.config.is_encoder_decoder = True
        self.model.model.decoder.embed_tokens.padding_idx = pad_token_id
        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_inference

        self.model.resize_token_embeddings(vocab_size)
        self.model.tie_weights()

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
        if past is not None:
            past_key_values = past

        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id).long()

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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.model.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.model.config.use_return_dict
        )

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
            loss = loss_fct(
				logits.reshape(-1, self.model.config.vocab_size),
				labels.reshape(-1),
			)

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


# ============================================================
# 3. Config
# ============================================================
class DonutConfig(PretrainedConfig):
    model_type = "invoice_donut"

    def __init__(
        self,
        input_size=(1280, 960),
        window_size=10,
        encoder_layer=(1, 1, 1, 1),
        max_position_embeddings=768,
        decoder_layer=1,
        vocab_size=57568,
        pad_token_id=1,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.input_size = list(input_size)
        self.window_size = window_size
        self.encoder_layer = list(encoder_layer)
        self.max_position_embeddings = max_position_embeddings
        self.decoder_layer = decoder_layer
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id


# ============================================================
# 4. Full Donut Model
# ============================================================
class DonutModel(PreTrainedModel):
    config_class = DonutConfig
    base_model_prefix = "donut"

    _keys_to_ignore_on_load_missing = [
        r"decoder\.model\.lm_head\.weight",
    ]

    def __init__(self, config: DonutConfig):
        super().__init__(config)

        self.encoder = SwinEncoder(
            input_size=config.input_size,
            window_size=config.window_size,
            encoder_layer=config.encoder_layer,
        )

        self.decoder = BARTDecoder(
            max_position_embeddings=config.max_position_embeddings,
            decoder_layer=config.decoder_layer,
            vocab_size=config.vocab_size,
            pad_token_id=config.pad_token_id,
        )

        self.post_init()

    def tie_weights(self, recompute_mapping: bool = True, **kwargs):
        if hasattr(self.decoder, "model") and hasattr(self.decoder.model, "tie_weights"):
            self.decoder.model.tie_weights()

    def forward(
        self,
        image_tensors: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_labels: torch.Tensor = None,
    ):
        encoder_outputs = self.encoder(image_tensors)
        encoder_hidden_states = encoder_outputs.flatten(1, 2)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            labels=decoder_labels,
        )
        return decoder_outputs
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        for m in model.modules():
            if hasattr(m, "init_non_persistent_buffers"):
                m.init_non_persistent_buffers()
        return model
    
    @torch.no_grad()
    def predict(
        self,
        image,
        tokenizer,
        input_size=[1280, 960],
        align_long_axis=False,
        random_padding=False,
        prompts="<s_cord-v2>",
        max_length=768,
    ):
        from .image_process import process_image, image_to_tensor
        from .inference import inference
        
        img = process_image(image, input_size, align_long_axis, random_padding)
        image_tensors = image_to_tensor(img)
        image_tensors = image_tensors.unsqueeze(0)
        image_tensors = image_tensors.to(self.device)
        
        prompt_tensors = tokenizer(
			prompts,
			add_special_tokens=False,
			truncation=True,
			return_tensors="pt",
		)["input_ids"].to(self.device)
        result = inference(self, tokenizer, image_tensors, prompt_tensors, max_length)
        return result
    
    @torch.no_grad()
    def predict_batch(
        self,
        image_tensors,
        tokenizer,
        prompts,
        max_length=768,
    ):
        from .inference import inference_batch

        if not isinstance(prompts, list):
            raise TypeError(
                f"prompts must be list[str], got {type(prompts)}"
            )

        if image_tensors.ndim != 4:
            raise ValueError(
                "image_tensors must have shape [B, C, H, W], "
                f"got {image_tensors.shape}"
            )

        batch_size = image_tensors.size(0)

        if len(prompts) != batch_size:
            raise ValueError(
                f"Number of prompts ({len(prompts)}) must match "
                f"batch size ({batch_size})"
            )

        image_tensors = image_tensors.to(self.device)

        prompt_tensors = tokenizer(
            prompts,
            add_special_tokens=False,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )["input_ids"].to(self.device)

        result = inference_batch(
            self,
            tokenizer,
            image_tensors,
            prompt_tensors,
            max_length,
        )

        return result
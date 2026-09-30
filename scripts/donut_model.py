import torch.nn as nn
import torch
from scripts.bart_decoder import BARTDecoder
from scripts.swin_encoder import SwinEncoder

class DonutModel(nn.Module):
    def __init__(
        self, 
        tokenizer,
        input_size, window_size, encoder_layer,
        max_position_embeddings, decoder_layer
    ):
        super().__init__()
        self.encoder = SwinEncoder(
            input_size=input_size,
            window_size=window_size,
            encoder_layer=encoder_layer,
        )    
        self.decoder = BARTDecoder(
            max_position_embeddings=max_position_embeddings,
            decoder_layer=decoder_layer,
            tokenizer=tokenizer
        )   
        
    def forward(self, image_tensors: torch.Tensor, decoder_input_ids: torch.Tensor, decoder_labels: torch.Tensor):
        encoder_outputs = self.encoder(image_tensors)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.flatten(1, 2),
            labels=decoder_labels,
        )
        return decoder_outputs        
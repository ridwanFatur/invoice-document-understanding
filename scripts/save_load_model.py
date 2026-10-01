import torch
import subprocess
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from .model import DonutModel, DonutConfig

def save_model(model, optimizer, configs, path):
    checkpoint = {
        "configs": configs,
    }

    if model is not None:
        checkpoint["model_state_dict"] = model.state_dict()

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)
    
def create_model(network_config):
    config = DonutConfig(
        input_size=network_config["input_size"],
        window_size=network_config["window_size"],
        encoder_layer=network_config["encoder_layer"],
        max_position_embeddings=network_config["max_position_embeddings"],
        decoder_layer=network_config["decoder_layer"],
        vocab_size=network_config["vocab_size"],
        pad_token_id=network_config["pad_token_id"],
    )    
    model = DonutModel(config)
    return model

def load_checkpoint(path, device):  
    if str(path).startswith("gs://"):
        print("Using Dictionary Input and GCS Checkpoint")
        local_checkpoint_path = Path("tensors") / Path(path).name
        subprocess.run(["gcloud", "storage", "cp", path, local_checkpoint_path],check=True,) 
    else:
        print("Using Dictionary Input and Local Checkpoint")
        local_checkpoint_path = Path(path)  
              
    checkpoint = torch.load(local_checkpoint_path, map_location=device)
    return checkpoint

def get_model(checkpoint, device, repo_id):
    # checkpoint is dictionary
    path = checkpoint.get("path")
    configs = checkpoint.get("configs")
    
    if path is not None:
        print(f"Get Model Path: {path}")
    if configs is not None:
        print(f"Get Model Configs: {configs}")
        
    tags = {}
    
    if path is None and configs is not None:
        # Create model, optimizer full completely
        print("Get Model: Create model, optimizer full completely")
        
        # Init Model
        model = create_model(configs["network"])
        model = model.to(device)
        
        # Init Optimizer
        optimizer = torch.optim.Adam(model.parameters())
        
        # Tags
        tags = {"type": "init-full"}
    elif path is not None and configs is None:
        # Continue training
        print("Get Model: Continue training")
        remote_checkpoint = load_checkpoint(path, device)
        
        # Use remote configs
        configs = remote_checkpoint["configs"]
        
        # Init Model
        model = create_model(configs["network"])
        model = model.to(device)
        
        # Load Model
        model.load_state_dict(remote_checkpoint["model_state_dict"])
        
        # Init Optimizer
        optimizer = torch.optim.Adam(model.parameters())
        
        # Load Optimizer
        optimizer.load_state_dict(remote_checkpoint["optimizer_state_dict"])
        
        # Tags
        tags = {"type": "continue-training", "from": path}
    elif path is not None and configs is not None:
        # Use loaded model, but create new optimizer
        print("Get Model: Use loaded model, but create new optimizer")
        remote_checkpoint = load_checkpoint(path, device)
        
        # Use and update remote configs
        remote_configs = remote_checkpoint["configs"]
        configs["network"] = remote_configs["network"]
        
        # Init Model
        model = create_model(configs["network"])
        model = model.to(device)
        
        # Load Model
        model.load_state_dict(remote_checkpoint["model_state_dict"])  
          
        # Init Optimizer
        optimizer = torch.optim.Adam(model.parameters())  
        
        # Tags
        tags = {"type": "use-loaded-model", "from": path}            
    else:
        raise ValueError("Invalid checkpoint")
        
    if repo_id:
        loaded_model = AutoModel.from_pretrained(
            repo_id,
            trust_remote_code=True,
        )
        model.load_state_dict(loaded_model.state_dict())
        
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {num_params:,}")
    
    return model, optimizer, configs, tags
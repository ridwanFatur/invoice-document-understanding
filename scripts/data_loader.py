from torch.utils.data import Dataset, DataLoader
from .image_process import process_image, image_to_tensor
from .ground_truth_process import json2token, process_ground_truth, get_train_inputs
from torch.utils.data import Subset

class DonutDataset(Dataset):
    def __init__(
        self, 
        dataset,
        tokenizer,
        
        # Image
        input_size = [1280, 960],
        align_long_axis = False,
        random_padding = True,

        # Ground Truth
        task_start_token = "<s_cord-v2>",
        prompt_end_token = "<s_cord-v2>",
        max_length = 768,
    ):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer

        # Image
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.random_padding = random_padding

        # Ground Truth
        self.task_start_token = task_start_token
        self.max_length = max_length
        self.prompt_end_token = prompt_end_token

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]

        # Image
        img = process_image(sample["image"], self.input_size, self.align_long_axis, self.random_padding)
        image_tensor = image_to_tensor(img)
        
        # Ground Truth
        ground_truth = sample["ground_truth"]
        gt_token_sequence, _ = process_ground_truth(self.tokenizer, ground_truth, task_start_token=self.task_start_token)
        prompt_end_token_id = self.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        input_ids, labels = get_train_inputs(self.tokenizer, gt_token_sequence, prompt_end_token_id, max_length=self.max_length)

        return image_tensor, input_ids, labels
    
def get_data_loaders(
    train_loader,
    test_loader,
    batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=False,
    subset_size=None,
):
    if subset_size:
        train_dataset = Subset(
            train_dataset, 
            range(subset_size)
        )
        test_dataset = Subset(
            test_dataset, 
            range(subset_size)
        ) 
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )    
    return train_loader, test_loader   
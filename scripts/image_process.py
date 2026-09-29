from torchvision.transforms.functional import resize, rotate
import numpy as np
from PIL import ImageOps
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def process_image(img, input_size, align_long_axis, random_padding):
    img = img.copy()
    if align_long_axis and (
        (input_size[0] > input_size[1] and img.width > img.height)
        or (input_size[0] < input_size[1] and img.width < img.height)
    ):
        img = rotate(img, angle=-90, expand=True)    

    img = resize(img, min(input_size))
    img.thumbnail((input_size[1], input_size[0]))
    
    delta_width = input_size[1] - img.width
    delta_height = input_size[0] - img.height
    
    if random_padding:
        pad_width = np.random.randint(low=0, high=delta_width + 1)
        pad_height = np.random.randint(low=0, high=delta_height + 1)
    else:
        pad_width = delta_width // 2
        pad_height = delta_height // 2
    padding = (
        pad_width,
        pad_height,
        delta_width - pad_width,
        delta_height - pad_height,
    )
    img = ImageOps.expand(img, padding)  
    return img

def image_to_tensor(img):
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )   
    return to_tensor(img)
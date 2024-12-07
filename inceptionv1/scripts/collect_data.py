from random import randint

import sys
sys.path.append("superposition/imagenet")

from utils.data import *

preprocess_and_save_images("superposition/imagenet/sample/val_images.tar.gz", "superposition/imagenet/sample/processed_tensors",max_items=4096)

# sampled_dataset = sample_and_save_dataset(sample_size=4096)
# print(f"Saved dataset size: {len(sampled_dataset)}")

# class_num=randint(0,999)
# sampled_dataset = sample_and_save_class_dataset(class_num,f'./superposition/imagenet/sample_{class_num}')
# print(f"Saved dataset size: {len(sampled_dataset)}")
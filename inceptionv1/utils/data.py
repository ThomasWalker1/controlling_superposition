import os
from datasets import load_dataset, Dataset
from torch.utils.data import IterableDataset
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from PIL import Image
import io
import tarfile

import sys
sys.path.append("inceptionv1")

from utils.classes import IMAGENET2012_CLASSES

SAVEDIR='./superposition/imagenet/sample'

def sample_and_save_dataset(split='validation', sample_size=100, save_dir=SAVEDIR):

    os.makedirs(save_dir, exist_ok=True)

    already_sampled=len(os.listdir(save_dir))-1
    
    dataset = load_dataset('imagenet-1k', split=split, streaming=True)

    sampled_data = []
    pbar=tqdm(total=sample_size)
    for i, example in enumerate(dataset):
        if i<already_sampled:
            continue
        if len(sampled_data) >= sample_size:
            break
        
        # Convert PIL Image to numpy array for easier saving
        img_array = np.array(example['image'])
        
        # Save image
        img_path = os.path.join(save_dir, f'image_{i}.png')
        Image.fromarray(img_array).save(img_path)
        
        # Store metadata
        sampled_data.append({
            'image_path': img_path,
            'label': example['label']
        })
        pbar.update(1)
    pbar.close()
    
    sampled_dataset = Dataset.from_list(sampled_data)
    sampled_dataset.save_to_disk(os.path.join(save_dir, 'dataset_metadata'))
    
    return sampled_dataset

def sample_and_save_class_dataset(class_num,save_dir,split='validation'):

    os.makedirs(save_dir, exist_ok=True)
    
    dataset = load_dataset('imagenet-1k', split=split, streaming=True)

    sampled_data = []
    pbar=tqdm(total=50,desc=get_class(int(class_num)).split(",")[0])
    for i, example in enumerate(dataset):
        if example['label']==class_num:
        
            # Convert PIL Image to numpy array for easier saving
            img_array = np.array(example['image'])
            
            # Save image
            img_path = os.path.join(save_dir, f'image_{i}.png')
            Image.fromarray(img_array).save(img_path)
            
            # Store metadata
            sampled_data.append({
                'image_path': img_path,
                'label': example['label']
            })
            pbar.update(1)
    pbar.close()
    
    sampled_dataset = Dataset.from_list(sampled_data)
    sampled_dataset.save_to_disk(os.path.join(save_dir, 'dataset_metadata'))
    
    return sampled_dataset

def load_saved_dataset(save_dir=SAVEDIR):

    loaded_dataset = Dataset.load_from_disk(os.path.join(save_dir, 'dataset_metadata'))
    
    return loaded_dataset

def reconstruct_images(dataset):

    images = [Image.open(item['image_path']) for item in dataset]
    return images


class ImageNetSampleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        """
        Custom PyTorch Dataset for ImageNet sample
        
        Args:
            dataset: Hugging Face dataset with image paths and labels
            transform: Optional image transformations
        """
        self.dataset = dataset
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.dataset[idx]['image_path']
        image = Image.open(image_path).convert('RGB')
        
        # Transform image
        image = self.transform(image)
        
        # Get label
        label = self.dataset[idx]['label']
        
        return image, label
    

def format_for_plotting(tensor):
    """
    Reverse the normalization process for visualization
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Clone the tensor to avoid modifying the original
    tensor = tensor.clone()
    
    # Denormalize each channel
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)

    tensor=tensor.permute(1,2,0)

    # Clip to valid range and convert to numpy
    #return np.clip(tensor.permute(1, 2, 0).numpy(), 0, 1)
    for k in range(3):
        tensor[:,k,...]=(tensor[:,k,...]-tensor[:,k,...].min())/(tensor[:,k,...].max()-tensor[:,k,...].min())
    return np.array(tensor)

def get_class(labels):
    if isinstance(labels,int):
        class_key=list(IMAGENET2012_CLASSES.keys())[labels]
        return IMAGENET2012_CLASSES[class_key]
    else:
        classes=[]
        for label in labels:
            classes.append(get_class(int(label)))
        return classes
    

class StreamingImageNet1k(IterableDataset):
    def __init__(self, data_stream, max_items):
        """
        Streaming Dataset for ImageNet-1k.

        Args:
            data_stream (callable): A generator function that yields (image, label).
            transform (callable, optional): Transformations to apply to the images.
        """
        self.data_stream = data_stream
        self.max_items = max_items
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __iter__(self):
        count=0
        for item in self.data_stream:
            if self.max_items is not None and count >= self.max_items:
                break
            try:
                # Load the image
                image = item['image']

                # Apply transformations if provided
                if self.transform:
                    image = self.transform(image)

                # Get the label
                label = item['label']

                yield image, label
                count+=1
            except Exception as e:
                print(f"Error processing item: {e}")
                continue



class TarGzDataset(Dataset):
    def __init__(self, max_items=None, tar_path="./superposition/imagenet/sample/val_images.tar.gz", transform=None):
        """
        PyTorch-compatible dataset that streams data from a `.tar.gz` file.

        Args:
            tar_path: Path to the `.tar.gz` file.
            transform: Transformations to apply to the images.
        """
        self.tar_path = tar_path
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.image_paths = []
        
        # Open the tar file and list image files

        self.tar = tarfile.open(self.tar_path, 'r:gz')
        for member in self.tar.getmembers():
            if member.isfile() and member.name.endswith(('.JPEG')):
                self.image_paths.append(member.name)
            if max_items is not None:
                if len(self.image_paths)>=max_items:
                    break

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Retrieve a batch of images and their corresponding labels from the tar file.

        Args:
            index: Single index or list of indices.
        """
        # If index is a list (batching), process each item in the batch
        if isinstance(index, list):
            images = []
            labels = []
            for idx in index:
                image, label = self._load_image_and_label(idx)
                images.append(image)
                labels.append(label)
                labels_tensor = torch.tensor(labels, dtype=torch.long)  # Ensure integer dtype
                images_tensor = torch.stack(images)  # Stack images into a single batch

            return {"image": images_tensor, "label": labels_tensor}

        # Otherwise, handle the case for a single index
        image, label = self._load_image_and_label(index)
        return {"image": image, "label": label}

    def _load_image_and_label(self, index):
        """
        Helper method to load an image and label by index.
        """
        # Open the tar file and extract the image
        # Get the specific member (image file) from the tar archive
        member = self.tar.getmember(self.image_paths[index])

        # Extract the file-like object
        f = self.tar.extractfile(member)

        # Open the image using PIL
        image = Image.open(f).convert('RGB')

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Extract the label based on the directory structure
        label = self._extract_label(self.image_paths[index])

        return image, label

    def _extract_label(self, path):
        """
        Extract label from the file path. Assumes folder structure like `label/image.jpg`.
        """
        return list(IMAGENET2012_CLASSES.keys()).index(path.split('_')[-1].split(".")[0])
    
    def __del__(self):
        """ Close the tar file when the object is deleted """
        self.tar.close()


def preprocess_and_save_images(tar_path, output_dir, max_items=None):
    os.makedirs(output_dir, exist_ok=True)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    saved=0
    pbar=tqdm(None,total=max_items)
    with tarfile.open(tar_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if max_items is not None:
                if saved>=max_items:
                    break
            if member.isfile():  # Only process files
                f = tar.extractfile(member)
                img = Image.open(f).convert("RGB")
                tensor = preprocess(img)
                file_name=f"{saved}_{member.name.split(".")[0].split("_")[-1]}.pt"
                output_path = os.path.join(output_dir, file_name)
                torch.save(tensor, output_path)
                saved+=1
                pbar.update()

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_dir):
        self.tensor_dir = tensor_dir
        self.tensor_files = os.listdir(tensor_dir)
    
    def __len__(self):
        return len(self.tensor_files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.tensor_dir, self.tensor_files[idx])
        image = torch.load(file_path,weights_only=True)
        label = self._extract_label(file_path)
        return image, label
    
    def _extract_label(self, path):
        return list(IMAGENET2012_CLASSES.keys()).index(path.split('_')[-1].split(".")[0])
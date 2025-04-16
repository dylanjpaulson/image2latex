import cv2
import re
from PIL import Image
from torch.utils.data import Dataset
import os

class DataLoader(Dataset):
    """
    Initializes the DataLoader object.

    Args:
        image_directory (str): Path to the directory containing the images.
        mapping_file (str): Path to the file containing the image-formula mappings.
        vocab_obj (Vocab): Vocab object for tokenizing the formulas.
        transform (callable, optional): A function/transform to apply on the images. Defaults to None.
        max_tokens (int, optional): Maximum number of tokens allowed in the tokenized formula. Defaults to 50.
    """
    def __init__(self, image_directory, mapping_file, vocab_obj, transform=None, max_tokens=50):
        self.image_directory = image_directory
        self.transform = transform
        self.vocab_obj = vocab_obj
        self.vocab_list = vocab_obj.vocab_list
        self.max_tokens = max_tokens

        # Save the paths of the images
        self.image_paths = []
        for image_name in os.listdir(image_directory):
            self.image_paths.append(os.path.join(image_directory, image_name))

        # Load the image-formula mapping from the .lst file
        self.im2formula = {}
        with open(mapping_file, 'r', encoding='windows-1252') as f:
            for line in f:
                # Get the first item of the line
                image_name = line.strip().split()[0]
                # Get everything except the first item of the line, and join them together
                formula = ' '.join(line.strip().split()[1:])

                # Add the image-formula mapping to the dictionary
                self.im2formula[image_name] = formula

        print('Number of images: {}'.format(len(self.image_paths)))
        print('Number of im2formula mappings: {}'.format(len(self.im2formula)))

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.im2formula)
    
    def __getitem__(self, index):
        """
        Returns a tuple of the image, the formula, and the image name at the specified index.

        Args:
            index (int): Index of the item to return.

        Returns:
            tuple: A tuple of the image, the formula, and the image name at the specified index.
        """
        image_names = list(self.im2formula.keys())
        image_name = image_names[index]

        # Load the image from a file path
        img = Image.open(os.path.join(self.image_directory, image_name))

        formula = self.im2formula[image_name]

        if self.transform:
            img = self.transform(img)
        
        # Tokenize the formula
        # tokenized_formula = self.vocab_obj.split_string_by_tokens(formula, len(self.vocab_list))
        
        return img, formula, image_name

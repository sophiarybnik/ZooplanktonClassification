
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


def get_adaptive_dropout(image_height, image_width, p=0.3):
    """
    CoarseDropout with hole sizes locked to half the image dimensions.
    """
    half_h = max(1, image_height // 2)
    half_w = max(1, image_width // 2)

    return A.CoarseDropout(
        num_holes_range=(1, 3),
        hole_height_range=(half_h, half_h), 
        hole_width_range=(half_w, half_w),  
        p=p
    )


class CustomDataset():
    """
    Loads and pre-processes data
    """
    def __init__(self,
                 cfg, 
                 df, image_size,
                 transform=None, 
                mode = "val"):
        self.root_dir = cfg.DATA_DIR
        self.df = df
        self.file_names = df['file_name'].values
        self.labels = df['label'].values
        self.image_size = image_size


        # Resize image
        default_transform = [
            A.Resize(self.image_size, self.image_size),
        ]

        # Augmentations (only for training)
        augmentations = [
            A.Rotate(p=0.6, limit=[-45, 45]),
            A.HorizontalFlip(p=0.6),
            get_adaptive_dropout(self.image_size,self.image_size)
            # A.CoarseDropout(num_holes_range=(1, 1),             
            #                 hole_height_range=(64, 64),  
            #                 hole_width_range=(64, 64),        
            #                 p=0.3)
        ]
                        

        if transform:
            # If user provides custom transform
            self.transform = transform
        elif mode == "train":
            # augment training data
            self.transform = A.Compose(default_transform + augmentations + [ToTensorV2()])
        else:
            self.transform = A.Compose(default_transform + [ToTensorV2()])


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = os.path.join(self.root_dir, self.file_names[idx])

        image = cv2.imread(file_path)

        # Handle missing / corrupt image
        if image is None:
            raise ValueError(f"Failed to read image: {file_path}")

        # Convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations        
        augmented = self.transform(image=image)
        image = augmented['image']

        # Normalize between 0 and 1- exclude 0 from range
        image = image/(self.image_size-1)

        return image, label
    


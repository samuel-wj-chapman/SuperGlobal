import torch
import os
import cv2
import numpy as np
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from model.CVNet_Rerank_model import CVNet_Rerank

# Custom dataset for satellite images
class SatelliteImageDataset(torch.utils.data.Dataset):
    """Dataset for satellite images with location and zoom info in filenames."""

    def __init__(self, images_dir):
        """
        Args:
            images_dir (str): Directory with satellite images
        """
        self.images_dir = images_dir
        self._scale_list = [1.0]  # Using a single scale by default
        
        # Mean and SD values in BGR order (same as the original example)
        self._mean = [0.406, 0.456, 0.485]
        self._sd = [0.225, 0.224, 0.229]
        
        # Get list of all image files
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Directory not found: {images_dir}")
            
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        if not self.image_files:
            raise ValueError(f"No PNG images found in {images_dir}")
            
        self.image_files.sort()  # Sort the files for consistent ordering
        print(f"Found {len(self.image_files)} images in {images_dir}")

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        im = im.transpose([2, 0, 1])
        # [0, 255] -> [0, 1]
        im = im / 255.0
        # Color normalization
        im = self._color_norm(im, self._mean, self._sd)
        return im
    
    def _color_norm(self, im, mean, std):
        """Performs color normalization."""
        for i in range(im.shape[0]):
            im[i] = im[i] - mean[i]
            im[i] = im[i] / std[i]
        return im

    def __getitem__(self, index):
        # Load the image
        img_path = os.path.join(self.images_dir, self.image_files[index])
        try:
            im = cv2.imread(img_path)
            if im is None:
                raise ValueError(f"Failed to load image: {img_path}")
                
            im_list = []
            # Process image at different scales
            for scale in self._scale_list:
                if scale == 1.0:
                    im_np = im.astype(np.float32, copy=False)
                    im_list.append(im_np)
                elif scale < 1.0:
                    im_resize = cv2.resize(im, dsize=(0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    im_np = im_resize.astype(np.float32, copy=False)
                    im_list.append(im_np)
                elif scale > 1.0:
                    im_resize = cv2.resize(im, dsize=(0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    im_np = im_resize.astype(np.float32, copy=False)
                    im_list.append(im_np)      
                else:
                    raise ValueError(f"Invalid scale: {scale}")
      
        except Exception as e:
            print(f'Error loading {img_path}: {e}')
            # Return a dummy value or re-raise
            raise

        for idx in range(len(im_list)):
            im_list[idx] = self._prepare_im(im_list[idx])
            
        return im_list, self.image_files[index]

    def __len__(self):
        return len(self.image_files)

@torch.no_grad()
def extract_features(model, data_loader, scale_list, batch_size=1):
    """
    Extract features from images while preserving their filenames
    """
    model.eval()  # Set model to evaluation mode
    
    features = {}  # Dictionary to store features with filenames as keys
    count = 0
    
    with torch.no_grad():
        for im_list, filename in tqdm(data_loader):
            if count % 100 == 0:
                print(f"Images Processed: {count}")
            count += 1
            
            try:
                for idx in range(len(im_list)):
                    im_list[idx] = im_list[idx].cuda()
                    desc = model.extract_global_descriptor(im_list[idx], True, True, True, scale_list)
                    if len(desc.shape) == 1:
                        desc.unsqueeze_(0)
                    desc = F.normalize(desc, p=2, dim=1)
                    
                    # Store the normalized descriptor with the filename as key
                    features[filename[0]] = desc.detach().cpu()  # Use [0] since batch size is 1
            except Exception as e:
                print(f"Error processing image {filename[0]}: {e}")
                continue
    
    # Verify features were extracted
    if not features:
        raise ValueError("No features were extracted. Please check the model and images.")
        
    return features

def verify_features(features, output_path):
    """Verify that features were properly saved by attempting to load them"""
    try:
        # First save the features
        torch.save(features, output_path)
        print(f"Features saved to {output_path}")
        
        # Then try to load them back
        loaded_features = torch.load(output_path)
        
        # Basic verification
        if len(loaded_features) != len(features):
            print(f"Warning: Number of loaded features ({len(loaded_features)}) doesn't match saved features ({len(features)})")
        else:
            print(f"Successfully verified {len(features)} feature vectors")
            
        # Check a random feature
        if features:
            key = next(iter(features.keys()))
            if not torch.allclose(features[key], loaded_features[key]):
                print(f"Warning: Loaded feature for {key} doesn't match the original")
            else:
                print(f"Sample feature for {key} verified correctly")
                
    except Exception as e:
        print(f"Error verifying features: {e}")
        raise
        
    return True

def main():
    parser = argparse.ArgumentParser(description='Extract features from satellite images')
    parser.add_argument('--weight', required=True,
                        help='Path to model weights')
    parser.add_argument('--depth', default=101, type=int,
                        help='Depth of ResNet')
    parser.add_argument('--images_dir', default='./vpair/images',
                        help='Directory with satellite images')
    parser.add_argument('--output', default='satellite_features.pth',
                        help='Output file to save features')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size')
    parser.add_argument('--scale_factor', default=3, type=int,
                        help='Scale factor for feature extraction')
    args = parser.parse_args()
    
    # Verify paths exist
    if not os.path.exists(args.weight):
        raise FileNotFoundError(f"Model weights file not found: {args.weight}")
        
    if not os.path.exists(args.images_dir):
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    try:
        model = CVNet_Rerank(args.depth, 2048, True)
        
        # Load weights
        weight = torch.load(args.weight)
        weight_new = {}
        for i, j in zip(weight['model_state'].keys(), weight['model_state'].values()):
            weight_new[i.replace('globalmodel', 'encoder_q')] = j
                
        mis_key = model.load_state_dict(weight_new, strict=False)
        model.cuda()
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise
    
    # Create dataset and dataloader
    try:
        dataset = SatelliteImageDataset(args.images_dir)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,  # Keep batch size 1 to preserve filename mapping
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
    except Exception as e:
        print(f"Error creating dataset: {e}")
        raise
    
    # Extract features
    try:
        features = extract_features(model, dataloader, args.scale_factor)
        print(f"Extracted features for {len(features)} images")
        
        # Add some debug info about a sample feature
        if features:
            key = next(iter(features.keys()))
            feat = features[key]
            print(f"Sample feature shape: {feat.shape}")
            print(f"Sample feature norm: {torch.norm(feat)}")
            print(f"Sample feature min/max: {torch.min(feat)}/{torch.max(feat)}")
        
        # Save and verify features
        verify_features(features, args.output)
        
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc() 
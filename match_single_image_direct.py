import torch
import numpy as np
import cv2
import os
import argparse
import random
from tqdm import tqdm
import torch.nn.functional as F
from matplotlib import pyplot as plt
from model.CVNet_Rerank_model import CVNet_Rerank

def load_features_safely(features_path):
    """Load features with error handling"""
    try:
        print(f"Loading features from {features_path}")
        features = torch.load(features_path)
        print(f"Successfully loaded features for {len(features)} images")
        return features
    except EOFError:
        print(f"Error: The features file {features_path} appears to be empty or corrupted.")
        print("Please run the extract_vpair_features.py script again to generate valid features.")
        return None
    except Exception as e:
        print(f"Error loading features: {e}")
        return None

def get_random_image(images_dir):
    """Get a random image from the directory"""
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Directory not found: {images_dir}")
        
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    if not image_files:
        raise ValueError(f"No PNG images found in {images_dir}")
    
    random_image = random.choice(image_files)
    print(f"Selected random image: {random_image}")
    return random_image

def prepare_image_for_model(image_path, mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]):
    """Load and prepare an image for the model"""
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Convert to float32
        img = img.astype(np.float32)
        
        # Prepare image: transpose dimensions [H,W,C] -> [C,H,W]
        img = img.transpose(2, 0, 1)
        
        # Normalize [0, 255] -> [0, 1]
        img = img / 255.0
        
        # Apply color normalization
        for i in range(img.shape[0]):
            img[i] = img[i] - mean[i]
            img[i] = img[i] / std[i]
            
        # Add batch dimension and convert to tensor
        img = torch.from_numpy(img).unsqueeze(0)
        
        return img
    except Exception as e:
        print(f"Error preparing image: {e}")
        raise

def extract_feature_from_image(model, img_tensor, scale_list=3):
    """Extract feature from a single image using the model"""
    try:
        model.eval()
        with torch.no_grad():
            img_tensor = img_tensor.cuda()
            # Extract descriptor
            desc = model.extract_global_descriptor(img_tensor, True, True, True, scale_list)
            if len(desc.shape) == 1:
                desc.unsqueeze_(0)
            # Normalize
            desc = F.normalize(desc, p=2, dim=1)
            
        return desc.detach().cpu()
    except Exception as e:
        print(f"Error extracting feature: {e}")
        raise

def match_image_against_features(query_feat, features, top_k=5):
    """Match query feature against all features in the database"""
    similarities = {}
    
    for img_name, feat in features.items():
        # Compute cosine similarity
        sim = torch.sum(query_feat * feat).item()
        similarities[img_name] = sim
    
    # Sort by similarity (descending)
    sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_matches[:top_k]

def visualize_matches(query_img_path, top_matches, features_dir, output_file=None):
    """Visualize the query image and its top matches"""
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Load and display query image
    query_img = cv2.imread(query_img_path)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    
    # Number of images to display (query + matches)
    n_images = len(top_matches) + 1
    
    # Plot query image
    plt.subplot(1, n_images, 1)
    plt.imshow(query_img)
    plt.title(f"Query: {os.path.basename(query_img_path)}")
    plt.axis('off')
    
    # Plot top matches
    for i, (match_name, similarity) in enumerate(top_matches):
        match_path = os.path.join(features_dir, match_name)
        if os.path.exists(match_path):
            match_img = cv2.imread(match_path)
            match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
            plt.subplot(1, n_images, i+2)
            plt.imshow(match_img)
            plt.title(f"Match {i+1}\n{match_name}\nScore: {similarity:.4f}")
            plt.axis('off')
        else:
            print(f"Warning: Match image file {match_path} not found")
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Process a single image through the model and match against database')
    parser.add_argument('--weight', required=True,
                        help='Path to model weights')
    parser.add_argument('--features', required=True,
                        help='Path to the extracted features file (.pth)')
    parser.add_argument('--images_dir', default='./vpair/images',
                        help='Directory containing the original images')
    parser.add_argument('--image', 
                        help='Specific image to process (if not provided, a random one will be selected)')
    parser.add_argument('--depth', default=101, type=int,
                        help='Depth of ResNet')
    parser.add_argument('--scale_factor', default=3, type=int,
                        help='Scale factor for feature extraction')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top matches to display')
    parser.add_argument('--output', default='match_result.png',
                        help='Output file for visualization')
    args = parser.parse_args()
    
    try:
        # Verify paths
        if not os.path.exists(args.weight):
            raise FileNotFoundError(f"Model weights file not found: {args.weight}")
            
        if not os.path.exists(args.images_dir):
            raise FileNotFoundError(f"Images directory not found: {args.images_dir}")
        
        # Load the model
        print("Loading model...")
        model = CVNet_Rerank(args.depth, 2048, True)
        
        # Load weights
        weight = torch.load(args.weight)
        weight_new = {}
        for i, j in zip(weight['model_state'].keys(), weight['model_state'].values()):
            weight_new[i.replace('globalmodel', 'encoder_q')] = j
                
        model.load_state_dict(weight_new, strict=False)
        model.cuda()
        model.eval()
        print("Model loaded successfully")
        
        # Load features
        features = load_features_safely(args.features)
        if features is None:
            return
        
        # Get image to process
        if args.image and os.path.exists(os.path.join(args.images_dir, args.image)):
            image_name = args.image
        else:
            # Select a random image from the directory
            image_name = get_random_image(args.images_dir)
        
        image_path = os.path.join(args.images_dir, image_name)
        
        print(f"Processing image: {image_path}")
        
        # Prepare the image for the model
        img_tensor = prepare_image_for_model(image_path)
        
        # Extract feature
        print("Extracting feature...")
        query_feat = extract_feature_from_image(model, img_tensor, args.scale_factor)
        print(f"Feature extracted, shape: {query_feat.shape}")
        print(f"Feature norm: {torch.norm(query_feat).item()}")
        
        # Match against the database
        print("Matching against database...")
        top_matches = match_image_against_features(query_feat, features, args.top_k)
        
        # Display results
        print(f"\nMatching results for {image_name}:")
        print("-" * 50)
        print(f"{'Image':<30} | {'Similarity':<10}")
        print("-" * 50)
        
        for img_name, sim in top_matches:
            print(f"{img_name:<30} | {sim:.6f}")
        
        # Check if the image itself is in the database
        if image_name in features:
            database_feat = features[image_name]
            self_similarity = torch.sum(query_feat * database_feat).item()
            print(f"\nSimilarity to precomputed feature: {self_similarity:.6f}")
            
            # Check if this is the highest score
            if top_matches[0][0] == image_name:
                print("✓ This image's precomputed feature is the closest match (as expected)")
            else:
                print(f"⚠ Warning: This image's precomputed feature is not the closest match!")
                print(f"  Closest match is {top_matches[0][0]} with score {top_matches[0][1]:.6f}")
        else:
            print(f"\nNote: Image {image_name} is not in the precomputed features database")
        
        # Visualize the results
        visualize_matches(image_path, top_matches, args.images_dir, args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
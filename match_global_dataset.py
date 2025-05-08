import torch
import numpy as np
import cv2
import os
import argparse
import random
import re
import math
import gc  # For garbage collection
import sys
import time  # For timing measurements
from tqdm import tqdm
import torch.nn.functional as F
from matplotlib import pyplot as plt
from model.CVNet_Rerank_model import CVNet_Rerank
import json

# Global timing variables to track performance
timing_stats = {
    'feature_extraction_times': [],
    'matching_times': [],
    'total_processing_times': []
}

def load_features_safely(features_path):
    """Load features with error handling"""
    try:
        start_time = time.time()
        print(f"Loading features from {features_path}")
        # Clear memory before loading large feature files
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        features = torch.load(features_path, map_location='cpu')  # Load to CPU first
        load_time = time.time() - start_time
        print(f"Successfully loaded features for {len(features)} images in {load_time:.2f} seconds")
        return features
    except EOFError:
        print(f"Error: The features file {features_path} appears to be empty or corrupted.")
        print("Please run the extract_vpair_features.py script again to generate valid features.")
        return None
    except Exception as e:
        print(f"Error loading features: {e}")
        return None

def get_random_image(images_dir):
    """Get a random image from the directory with its metadata file"""
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Directory not found: {images_dir}")
        
    # Find all jpg images
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    if not image_files:
        raise ValueError(f"No JPG images found in {images_dir}")
    
    # Filter to only include images that have metadata files
    valid_images = []
    for img_file in image_files:
        metadata_file = img_file.replace('.jpg', '_metadata.txt')
        if os.path.exists(os.path.join(images_dir, metadata_file)):
            valid_images.append(img_file)
    
    if not valid_images:
        raise ValueError(f"No images with matching metadata files found in {images_dir}")
    
    random_image = random.choice(valid_images)
    print(f"Selected random image: {random_image}")
    return random_image

def read_metadata(metadata_path):
    """Read latitude and longitude from metadata file with simplified format"""
    try:
        with open(metadata_path, 'r') as f:
            lines = f.readlines()
            
        # Extract latitude and longitude from the file
        lat_line = None
        lon_line = None
        
        for line in lines:
            if "Latitude" in line or "latitude" in line.lower():
                lat_line = line
            elif "Longitude" in line or "longitude" in line.lower():
                lon_line = line
        
        if not lat_line or not lon_line:
            raise ValueError(f"Could not find latitude/longitude in {metadata_path}")
        
        # Extract the numeric values - handle both formats:
        # "Latitude: 50.53951644897461" and "Ground Truth Latitude: 50.612667083740234"
        lat = float(lat_line.split(':')[1].strip())
        lon = float(lon_line.split(':')[1].strip())
        
        return lat, lon
    except Exception as e:
        print(f"Error reading metadata from {metadata_path}: {e}")
        raise

def extract_coordinates_from_filename(filename):
    """
    Extract latitude and longitude from a filename. Supports two formats:
    1. elevation_lat_lon.png (e.g., 512_50.61254_7.22798.png)
    2. zoom-x-y.png (e.g., 16-33987-21998.png)
    """
    try:
        # First try the elevation_lat_lon format
        match = re.search(r'(\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)', filename)
        if match:
            # Groups: 1=elevation, 2=latitude, 3=longitude
            elevation = float(match.group(1))
            lat = float(match.group(2))
            lon = float(match.group(3))
            return lat, lon
            
        # If that fails, try the zoom-x-y format
        match = re.search(r'(\d+)-(\d+)-(\d+)', filename)
        if match:
            # Groups: 1=zoom, 2=x, 3=y
            zoom = int(match.group(1))
            x = int(match.group(2))
            y = int(match.group(3))
            
            # Convert tile coordinates to lat/lon using Web Mercator projection
            # Based on: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
            lon = x / (2.0 ** zoom) * 360.0 - 180.0
            lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / (2.0 ** zoom))))
            lat = math.degrees(lat_rad)
            
            return lat, lon
            
        # If both formats fail, return None
        return None, None
        
    except Exception as e:
        print(f"Error extracting coordinates from filename {filename}: {e}")
        return None, None

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

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
        # Start timing for feature extraction
        start_time = time.time()
        
        # Clear CUDA cache before feature extraction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        model.eval()
        with torch.no_grad():
            # Handle CUDA out of memory errors by falling back to CPU if needed
            try:
                img_tensor = img_tensor.cuda()
                # Extract descriptor
                desc = model.extract_global_descriptor(img_tensor, True, True, True, scale_list)
                if len(desc.shape) == 1:
                    desc.unsqueeze_(0)
                # Normalize
                desc = F.normalize(desc, p=2, dim=1)
                
                result = desc.detach().cpu()
                
                # Record feature extraction time
                extraction_time = time.time() - start_time
                timing_stats['feature_extraction_times'].append(extraction_time)
                print(f"Feature extraction completed in {extraction_time:.3f} seconds")
                
                return result
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("CUDA out of memory error. Attempting to process on CPU...")
                    # Free GPU memory
                    torch.cuda.empty_cache()
                    
                    # Move model to CPU
                    model = model.cpu()
                    
                    # Process on CPU (this will be slower)
                    cpu_start_time = time.time()
                    desc = model.extract_global_descriptor(img_tensor, True, True, True, scale_list)
                    if len(desc.shape) == 1:
                        desc.unsqueeze_(0)
                    desc = F.normalize(desc, p=2, dim=1)
                    
                    result = desc.detach()
                    
                    # Record feature extraction time (CPU fallback)
                    extraction_time = time.time() - start_time
                    timing_stats['feature_extraction_times'].append(extraction_time)
                    print(f"Feature extraction (CPU fallback) completed in {extraction_time:.3f} seconds")
                    
                    return result
                else:
                    raise
    except Exception as e:
        print(f"Error extracting feature: {e}")
        raise

def match_image_against_features(query_feat, features, top_k=100):
    """Match query feature against all features in the database"""
    try:
        # Start timing for matching
        start_time = time.time()
        
        similarities = {}
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        feature_items = list(features.items())
        
        for i in range(0, len(feature_items), batch_size):
            batch_items = feature_items[i:i+batch_size]
            
            for img_name, feat in batch_items:
                # Compute cosine similarity
                sim = torch.sum(query_feat * feat).item()
                similarities[img_name] = sim
            
            # Clear memory after each batch
            if i % 5000 == 0 and i > 0:
                gc.collect()
        
        # Sort by similarity (descending)
        sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        matches = sorted_matches[:top_k]
        
        # Record matching time
        matching_time = time.time() - start_time
        timing_stats['matching_times'].append(matching_time)
        print(f"Feature matching completed in {matching_time:.3f} seconds for {len(feature_items)} images")
        print(f"Average time per 1000 images: {matching_time / (len(feature_items)/1000):.3f} seconds")
        
        return matches
    except Exception as e:
        print(f"Error in match_image_against_features: {e}")
        # Return empty list as fallback
        return []

def visualize_matches_with_distance(query_img_path, query_coords, top_matches, features_dir, output_file=None, max_vis=5):
    """Visualize the query image and its top matches with geographical distance"""
    try:
        # Create figure
        plt.figure(figsize=(20, 12))
        
        # Load and display query image
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # Number of images to display (query + matches)
        n_images = min(max_vis, len(top_matches) + 1)  # Limit to max_vis images total
        
        # Plot query image
        plt.subplot(1, n_images, 1)
        plt.imshow(query_img)
        plt.title(f"Query: {os.path.basename(query_img_path)}\nLat: {query_coords[0]:.6f}, Lon: {query_coords[1]:.6f}")
        plt.axis('off')
        
        # Plot top matches
        for i, (match_name, similarity, distance) in enumerate(top_matches[:n_images-1]):
            match_path = os.path.join(features_dir, match_name)
            if os.path.exists(match_path):
                try:
                    match_img = cv2.imread(match_path)
                    if match_img is not None:
                        match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
                        plt.subplot(1, n_images, i+2)
                        plt.imshow(match_img)
                        
                        # Extract coordinates for title
                        match_lat, match_lon = extract_coordinates_from_filename(match_name)
                        coord_text = f"Lat: {match_lat:.6f}, Lon: {match_lon:.6f}" if match_lat and match_lon else "No coords"
                        
                        plt.title(f"Match {i+1}: {os.path.basename(match_name)}\n{coord_text}\nSim: {similarity:.4f}, Dist: {distance:.2f} km")
                        plt.axis('off')
                    else:
                        print(f"Warning: Failed to read image {match_path}")
                except Exception as e:
                    print(f"Error processing match image {match_path}: {e}")
            else:
                print(f"Warning: Match image file {match_path} not found")
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            print(f"Visualization saved to {output_file}")
        else:
            plt.show()
        
        plt.close()
    except Exception as e:
        print(f"Error in visualization: {e}")
        # Just continue without visualization if there's an error

def save_results_to_csv(query_image, matches_with_distance, output_path):
    """Save matching results to CSV file for further analysis"""
    import csv
    
    try:
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['rank', 'image_name', 'similarity', 'distance_km', 'latitude', 'longitude']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            # Write query image info as rank 0
            writer.writerow({
                'rank': 0,
                'image_name': query_image,
                'similarity': 1.0,
                'distance_km': 0.0,
                'latitude': 'N/A',
                'longitude': 'N/A'
            })
            
            # Write all matches
            for rank, (img_name, similarity, distance) in enumerate(matches_with_distance, 1):
                match_lat, match_lon = extract_coordinates_from_filename(img_name)
                lat_str = f"{match_lat:.6f}" if match_lat is not None else "N/A"
                lon_str = f"{match_lon:.6f}" if match_lon is not None else "N/A"
                distance_str = f"{distance:.2f}" if not math.isnan(distance) else "N/A"
                
                writer.writerow({
                    'rank': rank,
                    'image_name': img_name,
                    'similarity': f"{similarity:.6f}",
                    'distance_km': distance_str,
                    'latitude': lat_str,
                    'longitude': lon_str
                })
        
        print(f"Results saved to CSV file: {output_path}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

def main():
    parser = argparse.ArgumentParser(description='Match images from global dataset and calculate geographical distance')
    parser.add_argument('--weight', required=True,
                        help='Path to model weights')
    parser.add_argument('--features', required=True,
                        help='Path to the extracted features file (.pth)')
    parser.add_argument('--global_dir', default='./global_dataset',
                        help='Directory containing global dataset images and metadata')
    parser.add_argument('--vpair_dir', default='./vpair/images',
                        help='Directory containing satellite images for matching')
    parser.add_argument('--image', 
                        help='Specific image name to process (if not provided, a random one will be selected)')
    parser.add_argument('--depth', default=101, type=int,
                        help='Depth of ResNet model')
    parser.add_argument('--scale_factor', default=3, type=int,
                        help='Scale factor for feature extraction')
    parser.add_argument('--top_k', type=int, default=100,
                        help='Number of top matches to return')
    parser.add_argument('--max_vis', type=int, default=5,
                        help='Maximum number of images to visualize')
    parser.add_argument('--output', default='global_match_result.png',
                        help='Output file for visualization')
    parser.add_argument('--csv_output', default='matching_results.csv',
                        help='Output CSV file for detailed results')
    parser.add_argument('--cpu_only', action='store_true',
                        help='Force CPU-only mode even if CUDA is available')
    parser.add_argument('--timing_output', default='',
                        help='Output file for timing statistics (optional)')
    parser.add_argument('--skip_visualization', action='store_true',
                        help='Skip generating visualization to save time')
    
    args = parser.parse_args()
    
    try:
        # Start total processing timer
        total_start_time = time.time()
        
        # Configure memory usage
        if torch.cuda.is_available() and not args.cpu_only:
            # Set memory settings to reduce fragmentation
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            # Print available CUDA memory for debugging
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f"Available CUDA memory: {free_mem / 1024**2:.1f}MB / {total_mem / 1024**2:.1f}MB")
        
        # Early notification if visualizations will be skipped
        if args.skip_visualization:
            print("Visualization will be skipped to save time")
        
        # Verify paths
        if not os.path.exists(args.weight):
            raise FileNotFoundError(f"Model weights file not found: {args.weight}")
            
        if not os.path.exists(args.global_dir):
            raise FileNotFoundError(f"Global dataset directory not found: {args.global_dir}")
            
        if not os.path.exists(args.vpair_dir):
            raise FileNotFoundError(f"VPair images directory not found: {args.vpair_dir}")
        
        # Clear memory before loading model
        gc.collect()
        if torch.cuda.is_available() and not args.cpu_only:
            torch.cuda.empty_cache()
        
        # Load the model
        model_load_start = time.time()
        print("Loading model...")
        device = torch.device('cpu' if args.cpu_only or not torch.cuda.is_available() else 'cuda')
        model = CVNet_Rerank(args.depth, 2048, True)
        
        # Load weights
        weight = torch.load(args.weight, map_location='cpu')  # Load to CPU first for safety
        weight_new = {}
        for i, j in zip(weight['model_state'].keys(), weight['model_state'].values()):
            weight_new[i.replace('globalmodel', 'encoder_q')] = j
                
        model.load_state_dict(weight_new, strict=False)
        
        if device.type == 'cuda':
            model = model.cuda()
        model.eval()
        model_load_time = time.time() - model_load_start
        print(f"Model loaded successfully in {model_load_time:.2f} seconds")
        
        # Load features
        features = load_features_safely(args.features)
        if features is None:
            return
        
        # Get image to process
        if args.image and os.path.exists(os.path.join(args.global_dir, args.image)):
            image_name = args.image
        else:
            # Select a random image from the directory
            image_name = get_random_image(args.global_dir)
        
        image_path = os.path.join(args.global_dir, image_name)
        metadata_path = os.path.join(args.global_dir, image_name.replace('.jpg', '_metadata.txt'))
        
        # Read metadata to get ground truth coordinates
        lat, lon = read_metadata(metadata_path)
        print(f"Image coordinates: Latitude {lat}, Longitude {lon}")
        
        print(f"Processing image: {image_path}")
        
        # Prepare the image for the model
        img_prep_start = time.time()
        img_tensor = prepare_image_for_model(image_path)
        img_prep_time = time.time() - img_prep_start
        print(f"Image preparation completed in {img_prep_time:.3f} seconds")
        
        # Extract feature
        print("Extracting feature...")
        try:
            query_feat = extract_feature_from_image(model, img_tensor, args.scale_factor)
            print(f"Feature extracted, shape: {query_feat.shape}")
        except Exception as e:
            print(f"Fatal error during feature extraction: {e}")
            sys.exit(1)
        
        # Free up GPU memory after feature extraction
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Match against the database
        print(f"Matching against database (top {args.top_k} matches)...")
        matches = match_image_against_features(query_feat, features, args.top_k)
        
        # Add geographical distance to matches and filter for valid matches with coordinates
        matches_with_distance = []
        for img_name, similarity in matches:
            match_lat, match_lon = extract_coordinates_from_filename(img_name)
            if match_lat is not None and match_lon is not None:
                distance = haversine_distance(lat, lon, match_lat, match_lon)
                matches_with_distance.append((img_name, similarity, distance))
            else:
                print(f"Warning: Could not extract coordinates from {img_name}")
                matches_with_distance.append((img_name, similarity, float('nan')))
        
        # Record total processing time
        total_processing_time = time.time() - total_start_time
        timing_stats['total_processing_times'].append(total_processing_time)
        
        # Display top 10 results in console
        print(f"\nTop 10 matching results for {image_name}:")
        print("-" * 80)
        print(f"{'Rank':<5} | {'Image':<40} | {'Similarity':<10} | {'Distance (km)':<15}")
        print("-" * 80)
        
        for i, (img_name, similarity, distance) in enumerate(matches_with_distance[:10]):
            distance_str = f"{distance:.2f}" if not math.isnan(distance) else "N/A"
            print(f"{i+1:<5} | {img_name:<40} | {similarity:.6f} | {distance_str:<15}")
        
        # Display timing information
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        print(f"Feature extraction time: {timing_stats['feature_extraction_times'][0]:.3f} seconds")
        print(f"Feature matching time: {timing_stats['matching_times'][0]:.3f} seconds")
        print(f"Total processing time: {total_processing_time:.3f} seconds")
        print("-"*50)
        
        # Save results to CSV
        print(f"\nSaving detailed results to {args.csv_output}...")
        save_results_to_csv(image_name, matches_with_distance, args.csv_output)
        
        # Visualize results (if not skipped)
        if not args.skip_visualization:
            print(f"Generating visualization...")
            visualize_matches_with_distance(image_path, (lat, lon), matches_with_distance, args.vpair_dir, args.output, args.max_vis)
        else:
            print("Visualization skipped as requested")
        
        # Save timing statistics if requested
        if args.timing_output:
            save_timing_stats(args.timing_output, timing_stats, len(features))
        
        print(f"Processing complete for {image_name}")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def save_timing_stats(output_file, timing_stats, num_features):
    """Save timing statistics to a JSON file"""
    try:
        # Create a structured dictionary for easy parsing
        timing_data = {
            'feature_extraction_time': timing_stats['feature_extraction_times'][0],
            'matching_time': timing_stats['matching_times'][0],
            'total_time': timing_stats['total_processing_times'][0],
            'num_features': num_features,
            'matching_time_per_1000': timing_stats['matching_times'][0] / (num_features/1000)
        }
        
        # Save as JSON
        with open(output_file, 'w') as f:
            json.dump(timing_data, f)
        
        print(f"Timing statistics saved to {output_file}")
    except Exception as e:
        print(f"Error saving timing statistics: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
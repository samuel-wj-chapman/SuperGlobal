import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import traceback

def load_features(features_path):
    """Load precomputed features from a .pth file"""
    try:
        print(f"Loading features from {features_path}")
        features = torch.load(features_path)
        print(f"Loaded {len(features)} feature vectors")
        
        # Verify at least one feature vector exists and is valid
        if features:
            key = next(iter(features.keys()))
            feat = features[key]
            print(f"Sample feature shape: {feat.shape}")
            print(f"Sample feature norm: {torch.norm(feat)}")
            
        return features
    except EOFError:
        print(f"Error: The features file {features_path} appears to be empty or corrupted.")
        print("Please run the extract_vpair_features.py script again to generate valid features.")
        return None
    except Exception as e:
        print(f"Error loading features: {e}")
        traceback.print_exc()
        return None

def compute_similarity_matrix(features):
    """Compute cosine similarity between all feature pairs"""
    print("Computing similarity matrix...")
    try:
        # Extract feature vectors into a single tensor
        filenames = list(features.keys())
        feature_matrix = torch.stack([features[name] for name in filenames], dim=0)
        
        # Compute cosine similarity (dot product between normalized vectors)
        similarity_matrix = torch.matmul(feature_matrix, feature_matrix.transpose(0, 1))
        
        return similarity_matrix, filenames
    except Exception as e:
        print(f"Error computing similarity matrix: {e}")
        traceback.print_exc()
        return None, []

def analyze_similarity_results(similarity_matrix, filenames):
    """Analyze the similarity matrix and return statistics"""
    print("Analyzing similarity results...")
    try:
        # Get the size of the similarity matrix
        n = similarity_matrix.shape[0]
        
        # For each query, find the most similar images and their scores
        top_scores = []
        self_match_scores = []
        
        results = []
        
        for i in range(n):
            query_name = filenames[i]
            
            # Get all similarity scores for this query
            scores = similarity_matrix[i].detach().cpu().numpy()
            
            # Record the self-match score (should be very close to 1.0)
            self_score = scores[i]
            self_match_scores.append(self_score)
            
            # Sort indices by score (descending)
            sorted_indices = np.argsort(-scores)
            
            # Record top match (excluding self)
            top_match_idx = sorted_indices[1]  # Skip the first, which should be self
            top_match_name = filenames[top_match_idx]
            top_match_score = scores[top_match_idx]
            top_scores.append(top_match_score)
            
            # Get all other scores (excluding self)
            other_indices = [j for j in range(n) if j != i]
            other_scores = scores[other_indices]
            
            # Calculate statistics
            results.append({
                'query': query_name,
                'self_match_score': self_score,
                'top_match': top_match_name,
                'top_match_score': top_match_score,
                'mean_score': np.mean(other_scores),
                'median_score': np.median(other_scores),
                'min_score': np.min(other_scores),
                'max_score': np.max(other_scores),
                'std_score': np.std(other_scores)
            })
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        return df, similarity_matrix, filenames
    except Exception as e:
        print(f"Error analyzing similarity results: {e}")
        traceback.print_exc()
        return None, None, []

def visualize_results(df, similarity_matrix, filenames, images_dir, output_dir):
    """Visualize the matching results"""
    print("Visualizing results...")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Plot histogram of self-match scores
        plt.figure(figsize=(10, 6))
        plt.hist(df['self_match_score'], bins=20)
        plt.title('Distribution of Self-Match Scores')
        plt.xlabel('Similarity Score')
        plt.ylabel('Count')
        plt.axvline(x=1.0, color='r', linestyle='--', label='Perfect Match (1.0)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'self_match_scores.png'))
        plt.close()
        
        # 2. Plot histogram of top match scores (excluding self)
        plt.figure(figsize=(10, 6))
        plt.hist(df['top_match_score'], bins=20)
        plt.title('Distribution of Top Match Scores (Excluding Self)')
        plt.xlabel('Similarity Score')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'top_match_scores.png'))
        plt.close()
        
        # 3. Create a heatmap of the similarity matrix for a random subset
        if len(filenames) > 20:
            # If more than 20 images, select a random subset
            subset_indices = np.random.choice(len(filenames), 20, replace=False)
        else:
            subset_indices = range(len(filenames))
        
        subset_filenames = [os.path.basename(filenames[i]) for i in subset_indices]
        subset_sim_matrix = similarity_matrix[subset_indices][:, subset_indices].detach().cpu().numpy()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(subset_sim_matrix, annot=False, xticklabels=subset_filenames, 
                    yticklabels=subset_filenames, cmap='viridis', vmin=0, vmax=1)
        plt.title('Similarity Matrix Heatmap (Random Subset)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'similarity_heatmap.png'))
        plt.close()
        
        # 4. Plot the distribution of all non-self similarity scores
        all_nonself_scores = []
        n = similarity_matrix.shape[0]
        for i in range(n):
            scores = similarity_matrix[i].detach().cpu().numpy()
            nonself_scores = [scores[j] for j in range(n) if j != i]
            all_nonself_scores.extend(nonself_scores)
        
        plt.figure(figsize=(10, 6))
        plt.hist(all_nonself_scores, bins=50)
        plt.title('Distribution of All Non-Self Similarity Scores')
        plt.xlabel('Similarity Score')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'all_nonself_scores.png'))
        plt.close()
        
        # 5. Visualize a few examples
        # Check if images directory exists
        if not os.path.exists(images_dir):
            print(f"Warning: Images directory {images_dir} not found. Skipping example match visualization.")
            return
            
        n_examples = min(5, len(filenames))
        for i in range(n_examples):
            query_idx = np.random.randint(0, len(filenames))
            query_file = filenames[query_idx]
            
            # Get similarity scores
            scores = similarity_matrix[query_idx].detach().cpu().numpy()
            
            # Get top 4 matches (including self)
            top_indices = np.argsort(-scores)[:4]
            top_scores = scores[top_indices]
            
            # Create visualization
            plt.figure(figsize=(15, 10))
            
            # Display query image
            query_path = os.path.join(images_dir, query_file)
            if os.path.exists(query_path):
                query_img = cv2.imread(query_path)
                if query_img is None:
                    print(f"Warning: Failed to read image {query_path}. Skipping this example.")
                    continue
                    
                query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
                plt.subplot(1, 5, 1)
                plt.imshow(query_img)
                plt.title('Query: ' + os.path.basename(query_file))
                plt.axis('off')
                
                # Display top matches
                for j, (idx, score) in enumerate(zip(top_indices, top_scores)):
                    match_file = filenames[idx]
                    match_path = os.path.join(images_dir, match_file)
                    if os.path.exists(match_path):
                        match_img = cv2.imread(match_path)
                        if match_img is None:
                            print(f"Warning: Failed to read image {match_path}. Skipping this match.")
                            continue
                            
                        match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
                        plt.subplot(1, 5, j+2)
                        plt.imshow(match_img)
                        plt.title(f'Match {j+1}\nScore: {score:.4f}')
                        plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'example_match_{i+1}.png'))
                plt.close()
            else:
                print(f"Warning: Image file {query_path} not found. Skipping this example.")
        
        # 6. Save the detailed results to CSV
        df.to_csv(os.path.join(output_dir, 'matching_results.csv'), index=False)
        print(f"Results saved to {output_dir}")
    except Exception as e:
        print(f"Error in visualization: {e}")
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Match satellite image features')
    parser.add_argument('--features', required=True,
                        help='Path to the extracted features file (.pth)')
    parser.add_argument('--images_dir', default='./vpair/images',
                        help='Directory containing the original images')
    parser.add_argument('--output_dir', default='./matching_results',
                        help='Directory to save visualization results')
    args = parser.parse_args()
    
    try:
        # Load features
        features = load_features(args.features)
        if features is None:
            print("Cannot proceed without valid features. Please fix the features file.")
            return
        
        # Compute similarity matrix
        similarity_matrix, filenames = compute_similarity_matrix(features)
        if similarity_matrix is None:
            print("Failed to compute similarity matrix. Cannot proceed.")
            return
        
        # Analyze results
        results_df, similarity_matrix, filenames = analyze_similarity_results(similarity_matrix, filenames)
        if results_df is None:
            print("Failed to analyze similarity results. Cannot proceed.")
            return
        
        # Display summary statistics
        print("\nSummary Statistics for Self-Match Scores:")
        print(results_df['self_match_score'].describe())
        
        print("\nSummary Statistics for Top Match Scores (Excluding Self):")
        print(results_df['top_match_score'].describe())
        
        # Create visualizations
        visualize_results(results_df, similarity_matrix, filenames, args.images_dir, args.output_dir)
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 
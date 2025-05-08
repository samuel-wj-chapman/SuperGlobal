import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob

def combine_csv_files(file_pattern, match_threshold=0.3):
    """
    Combine data from multiple CSV files into separate match and non-match lists.
    
    Args:
        file_pattern: Glob pattern to match CSV files or list of file paths
        match_threshold: Distance threshold for considering a match (in km)
    
    Returns:
        Two lists: match_scores and non_match_scores
    """
    # Handle both string patterns and lists of files
    if isinstance(file_pattern, list):
        files = file_pattern
    else:
        files = glob.glob(file_pattern)
        
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return None, None
    
    print(f"Processing {len(files)} files")
    
    combined_matches = []
    combined_non_matches = []
    processed_files = 0
    
    for file in files:
        print(f"Processing {file}...")
        try:
            # Read the CSV file
            df = pd.read_csv(file)
            
            # Check if file has the expected columns
            if 'similarity' not in df.columns or 'distance_km' not in df.columns:
                print(f"Warning: File {file} does not have the expected columns. Skipping.")
                continue
            
            # Skip the first row which is the reference image
            df = df.iloc[1:].copy()
            
            # Convert distance_km and similarity to numeric values
            df['distance_km'] = pd.to_numeric(df['distance_km'], errors='coerce')
            df['similarity'] = pd.to_numeric(df['similarity'], errors='coerce')
            
            # Drop rows with NaN values in either column
            df = df.dropna(subset=['distance_km', 'similarity'])
            
            # Create a new column to identify matches
            df['is_match'] = df['distance_km'] < match_threshold
            
            # Extract match and non-match scores
            match_scores = df[df['is_match']]['similarity'].tolist()
            non_match_scores = df[~df['is_match']]['similarity'].tolist()
            
            combined_matches.extend(match_scores)
            combined_non_matches.extend(non_match_scores)
            processed_files += 1
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}. Skipping.")
    
    print(f"Successfully processed {processed_files} files")
    
    if not combined_matches and not combined_non_matches:
        print("No valid data found in any of the files.")
        return None, None
        
    return combined_matches, combined_non_matches

def plot_combined_histogram_same_plot(match_scores, non_match_scores, num_bins=20, match_threshold=0.3, output_file=None):
    """
    Create a combined histogram of match and non-match scores on the same plot.
    
    Args:
        match_scores: List of similarity scores for matches
        non_match_scores: List of similarity scores for non-matches
        num_bins: Number of bins for the histogram
        match_threshold: Distance threshold used for classification
        output_file: Path to save the output plot (if None, just display)
    """
    if not match_scores and not non_match_scores:
        print("No data available to plot")
        return
    
    # Convert to numpy arrays
    match_scores = np.array(match_scores)
    non_match_scores = np.array(non_match_scores)
    
    # Get overall min and max for consistent binning
    all_scores = np.concatenate([match_scores, non_match_scores]) if match_scores.size > 0 and non_match_scores.size > 0 else \
                 match_scores if match_scores.size > 0 else non_match_scores
    
    min_score = np.min(all_scores)
    max_score = np.max(all_scores)
    
    # Create bins
    bins = np.linspace(min_score, max_score, num_bins + 1)
    
    # Create the figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot non-matches on primary y-axis if we have any
    if non_match_scores.size > 0:
        non_match_counts, non_match_bins, non_match_patches = ax1.hist(
            non_match_scores, bins=bins, alpha=0.6, color='red', label=f'Non-matches (≥{match_threshold}km): {len(non_match_scores)}'
        )
        ax1.set_ylabel('Count (Non-matches)', color='red')
    else:
        print("No non-match scores available")
        non_match_counts = []
    
    # Create a secondary y-axis for matches
    ax2 = ax1.twinx()
    
    # Plot matches on secondary y-axis if we have any
    if match_scores.size > 0:
        match_counts, match_bins, match_patches = ax2.hist(
            match_scores, bins=bins, alpha=0.7, color='green', label=f'Matches (<{match_threshold}km): {len(match_scores)}'
        )
        ax2.set_ylabel('Count (Matches)', color='green')
    else:
        print("No match scores available")
        match_counts = []
    
    # Set labels and title
    ax1.set_xlabel('Similarity Score')
    
    # Set title
    plt.title('Combined Similarity Score Distribution\nMatches vs Non-Matches Across All Files', fontsize=16)
    
    # Add grid
    ax1.grid(alpha=0.3)
    
    # Add statistics as text
    if match_scores.size > 0:
        match_avg = np.mean(match_scores)
        match_avg_str = f"{match_avg:.4f}"
    else:
        match_avg = 'N/A'
        match_avg_str = 'N/A'
        
    if non_match_scores.size > 0:
        non_match_avg = np.mean(non_match_scores)
        non_match_avg_str = f"{non_match_avg:.4f}"
    else:
        non_match_avg = 'N/A'
        non_match_avg_str = 'N/A'
    
    # Add text box with statistics
    stats_text = (
        f"Matches (<{match_threshold}km):\n"
        f"  Count: {len(match_scores)}\n"
        f"  Avg Score: {match_avg_str}\n\n"
        f"Non-matches (≥{match_threshold}km):\n"
        f"  Count: {len(non_match_scores)}\n"
        f"  Avg Score: {non_match_avg_str}\n\n"
        f"Score Range: {min_score:.4f} - {max_score:.4f}"
    )
    
    plt.figtext(0.15, 0.02, stats_text, ha="left", fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    # Add legends for both axes
    # Get handles and labels from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    
    # Combine them and create a legend
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Leave space at the bottom for the stats text
    
    # Save or display the plot
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_file, dpi=300)
        print(f"Combined plot saved to {output_file}")
    else:
        plt.show()
    
    # Return some statistics
    return {
        'num_matches': len(match_scores),
        'num_non_matches': len(non_match_scores),
        'match_avg_score': match_avg if isinstance(match_avg, float) else None,
        'non_match_avg_score': non_match_avg if isinstance(non_match_avg, float) else None,
        'min_score': min_score,
        'max_score': max_score
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a combined histogram from multiple CSV files")
    parser.add_argument("file_pattern", help="Glob pattern for CSV files (e.g., '/path/to/files/*.csv')")
    parser.add_argument("--bins", type=int, default=20, help="Number of bins for histogram")
    parser.add_argument("--threshold", type=float, default=0.3, help="Distance threshold for matches (km)")
    parser.add_argument("--output", help="Output file path for the combined plot (e.g., 'combined_histogram.png')")
    parser.add_argument("--exclude", help="Pattern to exclude certain files", default=None)
    
    args = parser.parse_args()
    
    # Handle file exclusion if specified
    if args.exclude:
        all_files = glob.glob(args.file_pattern)
        exclude_files = glob.glob(args.exclude)
        files_to_use = [f for f in all_files if f not in exclude_files]
        if not files_to_use:
            print(f"No files left after exclusion. Pattern matched {len(all_files)} files, but {len(exclude_files)} were excluded.")
            exit(1)
        print(f"Using {len(files_to_use)} files after excluding {len(exclude_files)} files.")
        # Replace the pattern with the specific list of files
        args.file_pattern = files_to_use
    
    # Combine data from all matching files
    match_scores, non_match_scores = combine_csv_files(args.file_pattern, args.threshold)
    
    if match_scores is not None and non_match_scores is not None:
        # Plot the combined histogram
        stats = plot_combined_histogram_same_plot(match_scores, non_match_scores, args.bins, args.threshold, args.output)
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total Matches (<{args.threshold}km): {stats['num_matches']}")
        print(f"Total Non-matches (≥{args.threshold}km): {stats['num_non_matches']}")
        
        if stats['match_avg_score'] is not None:
            print(f"Average Match Score: {stats['match_avg_score']:.4f}")
        else:
            print("Average Match Score: N/A")
            
        if stats['non_match_avg_score'] is not None:
            print(f"Average Non-match Score: {stats['non_match_avg_score']:.4f}")
        else:
            print("Average Non-match Score: N/A")
            
        print(f"Overall Score Range: {stats['min_score']:.4f} - {stats['max_score']:.4f}")
    else:
        print("No data available for analysis.") 
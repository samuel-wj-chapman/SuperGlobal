import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob
from collections import defaultdict

def analyze_rankings(file_pattern, match_threshold=0.3):
    """
    Analyze top-N rankings from multiple CSV files.
    
    Args:
        file_pattern: Glob pattern to match CSV files or list of file paths
        match_threshold: Distance threshold for considering a match (in km)
    
    Returns:
        Dictionary with statistics
    """
    # Handle both string patterns and lists of files
    if isinstance(file_pattern, list):
        files = file_pattern
    else:
        files = glob.glob(file_pattern)
        
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return None
    
    print(f"Processing {len(files)} files")
    
    # Statistics storage
    stats = {
        'total_files': 0,
        'files_with_matches': 0,
        'files_without_matches': 0,
        'top_1_with_match': [],
        'top_1_without_match': [],
        'top_5_with_match': [],
        'top_5_without_match': [],
        'top_20_with_match': [],
        'top_20_without_match': [],
        'top_100_with_match': [],
        'top_100_without_match': [],
        'match_ranks': [],
        'top1_is_match': 0,
        'top1_scores': []
    }
    
    # Process each file
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
            
            # Sort by similarity score (descending)
            df = df.sort_values('similarity', ascending=False).reset_index(drop=True)
            
            # Find if there are any matches in the dataset
            has_match = df['is_match'].any()
            
            # Track file statistics
            stats['total_files'] += 1
            
            # Check if top 1 is a match
            if len(df) > 0:
                top1_score = df.iloc[0]['similarity']
                stats['top1_scores'].append(top1_score)
                if df.iloc[0]['is_match']:
                    stats['top1_is_match'] += 1
            
            if has_match:
                stats['files_with_matches'] += 1
                
                # Find the rank of the first match
                first_match_rank = df[df['is_match']].index.min() + 1  # +1 because index is 0-based
                stats['match_ranks'].append(first_match_rank)
                
                # Get scores for top N
                top_1_scores = df.iloc[:1]['similarity'].tolist()
                top_5_scores = df.iloc[:5]['similarity'].tolist()
                top_20_scores = df.iloc[:20]['similarity'].tolist()
                top_100_scores = df.iloc[:100]['similarity'].tolist() if len(df) >= 100 else df['similarity'].tolist()
                
                stats['top_1_with_match'].append(np.mean(top_1_scores))
                stats['top_5_with_match'].append(np.mean(top_5_scores))
                stats['top_20_with_match'].append(np.mean(top_20_scores))
                stats['top_100_with_match'].append(np.mean(top_100_scores))
            else:
                stats['files_without_matches'] += 1
                
                # Get scores for top N
                top_1_scores = df.iloc[:1]['similarity'].tolist()
                top_5_scores = df.iloc[:5]['similarity'].tolist()
                top_20_scores = df.iloc[:20]['similarity'].tolist()
                top_100_scores = df.iloc[:100]['similarity'].tolist() if len(df) >= 100 else df['similarity'].tolist()
                
                stats['top_1_without_match'].append(np.mean(top_1_scores))
                stats['top_5_without_match'].append(np.mean(top_5_scores))
                stats['top_20_without_match'].append(np.mean(top_20_scores))
                stats['top_100_without_match'].append(np.mean(top_100_scores))
                
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}. Skipping.")
    
    print(f"Successfully processed {stats['total_files']} files")
    
    # Calculate average statistics
    if stats['files_with_matches'] > 0:
        stats['avg_top_1_with_match'] = np.mean(stats['top_1_with_match'])
        stats['avg_top_5_with_match'] = np.mean(stats['top_5_with_match'])
        stats['avg_top_20_with_match'] = np.mean(stats['top_20_with_match'])
        stats['avg_top_100_with_match'] = np.mean(stats['top_100_with_match'])
        stats['avg_match_rank'] = np.mean(stats['match_ranks'])
        stats['median_match_rank'] = np.median(stats['match_ranks'])
    else:
        stats['avg_top_1_with_match'] = 'N/A'
        stats['avg_top_5_with_match'] = 'N/A'
        stats['avg_top_20_with_match'] = 'N/A'
        stats['avg_top_100_with_match'] = 'N/A'
        stats['avg_match_rank'] = 'N/A'
        stats['median_match_rank'] = 'N/A'
        
    if stats['files_without_matches'] > 0:
        stats['avg_top_1_without_match'] = np.mean(stats['top_1_without_match'])
        stats['avg_top_5_without_match'] = np.mean(stats['top_5_without_match'])
        stats['avg_top_20_without_match'] = np.mean(stats['top_20_without_match'])
        stats['avg_top_100_without_match'] = np.mean(stats['top_100_without_match'])
    else:
        stats['avg_top_1_without_match'] = 'N/A'
        stats['avg_top_5_without_match'] = 'N/A'
        stats['avg_top_20_without_match'] = 'N/A'
        stats['avg_top_100_without_match'] = 'N/A'
    
    # Calculate top 1 match rate
    stats['top1_match_rate'] = (stats['top1_is_match'] / stats['total_files']) * 100 if stats['total_files'] > 0 else 0
    
    # Success rates
    top_n_success = defaultdict(int)
    for rank in stats['match_ranks']:
        for n in [1, 5, 10, 20, 50, 100]:
            if rank <= n:
                top_n_success[n] += 1
    
    # Calculate success rates as percentages
    if stats['files_with_matches'] > 0:
        for n in [1, 5, 10, 20, 50, 100]:
            stats[f'success_rate_top_{n}'] = (top_n_success[n] / stats['total_files']) * 100
    
    return stats

def print_statistics(stats, threshold):
    """
    Print detailed statistics from the analysis
    """
    if not stats:
        print("No statistics available.")
        return
    
    print("\n====== RETRIEVAL STATISTICS ======")
    print(f"Total files processed: {stats['total_files']}")
    print(f"Files with at least one match (<{threshold}km): {stats['files_with_matches']} ({stats['files_with_matches']/stats['total_files']*100:.2f}%)")
    print(f"Files without matches: {stats['files_without_matches']} ({stats['files_without_matches']/stats['total_files']*100:.2f}%)")
    print(f"Top 1 is a match: {stats['top1_is_match']} ({stats['top1_match_rate']:.2f}%)")
    
    print("\n----- Average Similarity Scores -----")
    
    # Format the output based on whether we have data
    top1_with = f"{stats['avg_top_1_with_match']:.5f}" if isinstance(stats['avg_top_1_with_match'], float) else stats['avg_top_1_with_match']
    top5_with = f"{stats['avg_top_5_with_match']:.5f}" if isinstance(stats['avg_top_5_with_match'], float) else stats['avg_top_5_with_match']
    top20_with = f"{stats['avg_top_20_with_match']:.5f}" if isinstance(stats['avg_top_20_with_match'], float) else stats['avg_top_20_with_match']
    top100_with = f"{stats['avg_top_100_with_match']:.5f}" if isinstance(stats['avg_top_100_with_match'], float) else stats['avg_top_100_with_match']
    
    top1_without = f"{stats['avg_top_1_without_match']:.5f}" if isinstance(stats['avg_top_1_without_match'], float) else stats['avg_top_1_without_match']
    top5_without = f"{stats['avg_top_5_without_match']:.5f}" if isinstance(stats['avg_top_5_without_match'], float) else stats['avg_top_5_without_match']
    top20_without = f"{stats['avg_top_20_without_match']:.5f}" if isinstance(stats['avg_top_20_without_match'], float) else stats['avg_top_20_without_match']
    top100_without = f"{stats['avg_top_100_without_match']:.5f}" if isinstance(stats['avg_top_100_without_match'], float) else stats['avg_top_100_without_match']
    
    print(f"                   | When matches exist | When no matches")
    print(f"Top 1 average:     | {top1_with:16} | {top1_without}")
    print(f"Top 5 average:     | {top5_with:16} | {top5_without}")
    print(f"Top 20 average:    | {top20_with:16} | {top20_without}")
    print(f"Top 100 average:   | {top100_with:16} | {top100_without}")
    
    print("\n----- Match Ranking Statistics -----")
    if isinstance(stats['avg_match_rank'], float):
        print(f"Average rank of first match: {stats['avg_match_rank']:.2f}")
        print(f"Median rank of first match: {stats['median_match_rank']:.2f}")
    else:
        print(f"Average rank of first match: {stats['avg_match_rank']}")
        print(f"Median rank of first match: {stats['median_match_rank']}")
    
    print("\n----- Success Rates -----")
    for n in [1, 5, 10, 20, 50, 100]:
        if f'success_rate_top_{n}' in stats:
            print(f"Top {n:3} success rate: {stats[f'success_rate_top_{n}']:.2f}%")
    
    print("================================")

def plot_match_ranks(stats, output_file=None):
    """
    Create a histogram of match ranks
    """
    if not stats or 'match_ranks' not in stats or not stats['match_ranks']:
        print("No match rank data available for plotting.")
        return
    
    match_ranks = stats['match_ranks']
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Create bins that focus on lower ranks but also show the distribution
    bins = list(range(1, 21)) + list(range(20, 101, 5)) + list(range(100, max(match_ranks) + 50, 50))
    bins = [b for b in bins if b <= max(match_ranks) + 50]
    
    # Plot histogram
    plt.hist(match_ranks, bins=bins, alpha=0.7, color='blue')
    
    # Add lines for mean and median
    plt.axvline(stats['avg_match_rank'], color='red', linestyle='dashed', linewidth=2, label=f"Mean: {stats['avg_match_rank']:.2f}")
    plt.axvline(stats['median_match_rank'], color='green', linestyle='dashed', linewidth=2, label=f"Median: {stats['median_match_rank']:.2f}")
    
    # Add labels and title
    plt.xlabel('Rank of First Match')
    plt.ylabel('Count')
    plt.title('Distribution of First Match Ranks')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Add success rate annotations
    success_text = "Success Rates:\n"
    success_text += f"Top 1: {stats['top1_match_rate']:.2f}%\n"
    for n in [5, 10, 20, 50, 100]:
        if f'success_rate_top_{n}' in stats:
            success_text += f"Top {n}: {stats[f'success_rate_top_{n}']:.2f}%\n"
    
    plt.annotate(success_text, xy=(0.02, 0.97), xycoords='axes fraction',
                 va='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save or display the plot
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_file, dpi=300)
        print(f"Rank distribution plot saved to {output_file}")
    else:
        plt.show()

def plot_top1_score_distribution(stats, output_file=None):
    """
    Create a histogram of top 1 scores, comparing matches vs non-matches
    """
    if not stats or 'top1_scores' not in stats or not stats['top1_scores']:
        print("No top 1 score data available for plotting.")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Get top 1 scores
    top1_scores = np.array(stats['top1_scores'])
    
    # Create bins
    bins = np.linspace(min(top1_scores), max(top1_scores), 30)
    
    # Plot histogram
    plt.hist(top1_scores, bins=bins, alpha=0.7, color='purple')
    
    # Add labels and title
    plt.xlabel('Top 1 Similarity Score')
    plt.ylabel('Count')
    plt.title('Distribution of Top 1 Similarity Scores')
    plt.grid(alpha=0.3)
    
    # Add match rate annotation
    plt.annotate(f"Top 1 Match Rate: {stats['top1_match_rate']:.2f}%", 
                 xy=(0.02, 0.97), xycoords='axes fraction',
                 va='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save or display the plot
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_file, dpi=300)
        print(f"Top 1 score distribution plot saved to {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ranking statistics from image retrieval results")
    parser.add_argument("file_pattern", help="Glob pattern for CSV files (e.g., '/path/to/files/*.csv')")
    parser.add_argument("--threshold", type=float, default=0.3, help="Distance threshold for matches (km)")
    parser.add_argument("--output", help="Output directory for plots", default=None)
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
    
    # Analyze ranking statistics
    stats = analyze_rankings(args.file_pattern, args.threshold)
    
    if stats:
        # Print detailed statistics
        print_statistics(stats, args.threshold)
        
        # Plot match rank distribution if we have match data
        if args.output and stats['files_with_matches'] > 0:
            # Create the output directory if it doesn't exist
            os.makedirs(args.output, exist_ok=True)
            
            # Save the match rank distribution plot
            rank_plot_file = os.path.join(args.output, "match_rank_distribution.png")
            plot_match_ranks(stats, rank_plot_file)
            
            # Save the top 1 score distribution plot
            top1_plot_file = os.path.join(args.output, "top1_score_distribution.png")
            plot_top1_score_distribution(stats, top1_plot_file)
        elif stats['files_with_matches'] > 0:
            plot_match_ranks(stats)
            plot_top1_score_distribution(stats)
    else:
        print("No data available for analysis.") 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob
from collections import defaultdict

def analyze_rankings_by_top_n(file_pattern, match_threshold=0.3):
    """
    Analyze rankings with statistics split by whether there are matches in specific top N cutoffs.
    
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
        
        # Scores for top 1
        'top_1_match_in_top1': [],      # Scores when match is in top 1
        'top_1_match_not_in_top1': [],  # Scores when match is not in top 1
        
        # Scores for top 5
        'top_5_match_in_top5': [],      # Scores when match is in top 5
        'top_5_match_not_in_top5': [],  # Scores when match is not in top 5
        
        # Scores for top 20
        'top_20_match_in_top20': [],    # Scores when match is in top 20
        'top_20_match_not_in_top20': [], # Scores when match is not in top 20
        
        # Scores for top 100
        'top_100_match_in_top100': [],   # Scores when match is in top 100
        'top_100_match_not_in_top100': [], # Scores when match is not in top 100
        
        # Counters for files with matches in top N
        'files_with_match_in_top1': 0,
        'files_with_match_in_top5': 0,
        'files_with_match_in_top20': 0,
        'files_with_match_in_top100': 0,
        
        # Match rank tracking
        'match_ranks': [],
        
        # Top 1 score tracking (separated by match status)
        'top1_scores_match': [],     # Top 1 scores when top 1 is a match
        'top1_scores_no_match': []   # Top 1 scores when top 1 is not a match
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
            
            # Get top 1 score and check if it's a match
            if len(df) > 0:
                top1_score = df.iloc[0]['similarity']
                top1_is_match = df.iloc[0]['is_match']
                
                # Store top 1 score based on whether it's a match
                if top1_is_match:
                    stats['top1_scores_match'].append(top1_score)
                else:
                    stats['top1_scores_no_match'].append(top1_score)
            
            if has_match:
                stats['files_with_matches'] += 1
                
                # Find the rank of the first match
                first_match_rank = df[df['is_match']].index.min() + 1  # +1 because index is 0-based
                stats['match_ranks'].append(first_match_rank)
                
                # Check if match is in top N
                match_in_top1 = first_match_rank <= 1
                match_in_top5 = first_match_rank <= 5
                match_in_top20 = first_match_rank <= 20
                match_in_top100 = first_match_rank <= 100
                
                # Update counters
                if match_in_top1:
                    stats['files_with_match_in_top1'] += 1
                if match_in_top5:
                    stats['files_with_match_in_top5'] += 1
                if match_in_top20:
                    stats['files_with_match_in_top20'] += 1
                if match_in_top100:
                    stats['files_with_match_in_top100'] += 1
                
                # Get average scores for top N
                top_1_scores = df.iloc[:1]['similarity'].mean()
                top_5_scores = df.iloc[:5]['similarity'].mean()
                top_20_scores = df.iloc[:20]['similarity'].mean()
                top_100_scores = df.iloc[:100]['similarity'].mean() if len(df) >= 100 else df['similarity'].mean()
                
                # Store scores based on whether match is in top N
                if match_in_top1:
                    stats['top_1_match_in_top1'].append(top_1_scores)
                else:
                    stats['top_1_match_not_in_top1'].append(top_1_scores)
                    
                if match_in_top5:
                    stats['top_5_match_in_top5'].append(top_5_scores)
                else:
                    stats['top_5_match_not_in_top5'].append(top_5_scores)
                    
                if match_in_top20:
                    stats['top_20_match_in_top20'].append(top_20_scores)
                else:
                    stats['top_20_match_not_in_top20'].append(top_20_scores)
                    
                if match_in_top100:
                    stats['top_100_match_in_top100'].append(top_100_scores)
                else:
                    stats['top_100_match_not_in_top100'].append(top_100_scores)
                
            else:
                stats['files_without_matches'] += 1
                
                # For files without any matches, we'll count them as not having matches in any top N
                # Get average scores
                top_1_scores = df.iloc[:1]['similarity'].mean()
                top_5_scores = df.iloc[:5]['similarity'].mean()
                top_20_scores = df.iloc[:20]['similarity'].mean()
                top_100_scores = df.iloc[:100]['similarity'].mean() if len(df) >= 100 else df['similarity'].mean()
                
                # Store in the "not in top N" categories
                stats['top_1_match_not_in_top1'].append(top_1_scores)
                stats['top_5_match_not_in_top5'].append(top_5_scores)
                stats['top_20_match_not_in_top20'].append(top_20_scores)
                stats['top_100_match_not_in_top100'].append(top_100_scores)
                
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}. Skipping.")
    
    print(f"Successfully processed {stats['total_files']} files")
    
    # Calculate average statistics
    
    # Top 1 statistics
    if stats['files_with_match_in_top1'] > 0:
        stats['avg_top_1_match_in_top1'] = np.mean(stats['top_1_match_in_top1'])
    else:
        stats['avg_top_1_match_in_top1'] = 'N/A'
        
    if len(stats['top_1_match_not_in_top1']) > 0:
        stats['avg_top_1_match_not_in_top1'] = np.mean(stats['top_1_match_not_in_top1'])
    else:
        stats['avg_top_1_match_not_in_top1'] = 'N/A'
    
    # Top 5 statistics
    if stats['files_with_match_in_top5'] > 0:
        stats['avg_top_5_match_in_top5'] = np.mean(stats['top_5_match_in_top5'])
    else:
        stats['avg_top_5_match_in_top5'] = 'N/A'
        
    if len(stats['top_5_match_not_in_top5']) > 0:
        stats['avg_top_5_match_not_in_top5'] = np.mean(stats['top_5_match_not_in_top5'])
    else:
        stats['avg_top_5_match_not_in_top5'] = 'N/A'
    
    # Top 20 statistics
    if stats['files_with_match_in_top20'] > 0:
        stats['avg_top_20_match_in_top20'] = np.mean(stats['top_20_match_in_top20'])
    else:
        stats['avg_top_20_match_in_top20'] = 'N/A'
        
    if len(stats['top_20_match_not_in_top20']) > 0:
        stats['avg_top_20_match_not_in_top20'] = np.mean(stats['top_20_match_not_in_top20'])
    else:
        stats['avg_top_20_match_not_in_top20'] = 'N/A'
    
    # Top 100 statistics
    if stats['files_with_match_in_top100'] > 0:
        stats['avg_top_100_match_in_top100'] = np.mean(stats['top_100_match_in_top100'])
    else:
        stats['avg_top_100_match_in_top100'] = 'N/A'
        
    if len(stats['top_100_match_not_in_top100']) > 0:
        stats['avg_top_100_match_not_in_top100'] = np.mean(stats['top_100_match_not_in_top100'])
    else:
        stats['avg_top_100_match_not_in_top100'] = 'N/A'
    
    # Match rank statistics
    if stats['files_with_matches'] > 0:
        stats['avg_match_rank'] = np.mean(stats['match_ranks'])
        stats['median_match_rank'] = np.median(stats['match_ranks'])
    else:
        stats['avg_match_rank'] = 'N/A'
        stats['median_match_rank'] = 'N/A'
    
    # Success rates
    stats['top1_success_rate'] = (stats['files_with_match_in_top1'] / stats['total_files']) * 100 if stats['total_files'] > 0 else 0
    stats['top5_success_rate'] = (stats['files_with_match_in_top5'] / stats['total_files']) * 100 if stats['total_files'] > 0 else 0
    stats['top20_success_rate'] = (stats['files_with_match_in_top20'] / stats['total_files']) * 100 if stats['total_files'] > 0 else 0
    stats['top100_success_rate'] = (stats['files_with_match_in_top100'] / stats['total_files']) * 100 if stats['total_files'] > 0 else 0
    
    # Calculate mean top 1 scores
    if len(stats['top1_scores_match']) > 0:
        stats['avg_top1_score_match'] = np.mean(stats['top1_scores_match'])
    else:
        stats['avg_top1_score_match'] = 'N/A'
        
    if len(stats['top1_scores_no_match']) > 0:
        stats['avg_top1_score_no_match'] = np.mean(stats['top1_scores_no_match'])
    else:
        stats['avg_top1_score_no_match'] = 'N/A'
    
    return stats

def print_statistics_by_top_n(stats, threshold):
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
    
    print("\n----- Success Rates -----")
    print(f"Top 1 success rate:   {stats['top1_success_rate']:.2f}% ({stats['files_with_match_in_top1']} files)")
    print(f"Top 5 success rate:   {stats['top5_success_rate']:.2f}% ({stats['files_with_match_in_top5']} files)")
    print(f"Top 20 success rate:  {stats['top20_success_rate']:.2f}% ({stats['files_with_match_in_top20']} files)")
    print(f"Top 100 success rate: {stats['top100_success_rate']:.2f}% ({stats['files_with_match_in_top100']} files)")
    
    print("\n----- Average Similarity Scores -----")
    
    # Format the output based on whether we have data
    top1_with = f"{stats['avg_top_1_match_in_top1']:.5f}" if isinstance(stats['avg_top_1_match_in_top1'], float) else stats['avg_top_1_match_in_top1']
    top5_with = f"{stats['avg_top_5_match_in_top5']:.5f}" if isinstance(stats['avg_top_5_match_in_top5'], float) else stats['avg_top_5_match_in_top5']
    top20_with = f"{stats['avg_top_20_match_in_top20']:.5f}" if isinstance(stats['avg_top_20_match_in_top20'], float) else stats['avg_top_20_match_in_top20']
    top100_with = f"{stats['avg_top_100_match_in_top100']:.5f}" if isinstance(stats['avg_top_100_match_in_top100'], float) else stats['avg_top_100_match_in_top100']
    
    top1_without = f"{stats['avg_top_1_match_not_in_top1']:.5f}" if isinstance(stats['avg_top_1_match_not_in_top1'], float) else stats['avg_top_1_match_not_in_top1']
    top5_without = f"{stats['avg_top_5_match_not_in_top5']:.5f}" if isinstance(stats['avg_top_5_match_not_in_top5'], float) else stats['avg_top_5_match_not_in_top5']
    top20_without = f"{stats['avg_top_20_match_not_in_top20']:.5f}" if isinstance(stats['avg_top_20_match_not_in_top20'], float) else stats['avg_top_20_match_not_in_top20']
    top100_without = f"{stats['avg_top_100_match_not_in_top100']:.5f}" if isinstance(stats['avg_top_100_match_not_in_top100'], float) else stats['avg_top_100_match_not_in_top100']
    
    print(f"                   | When match in top N | When match not in top N")
    print(f"Top 1 average:     | {top1_with:16} | {top1_without}")
    print(f"Top 5 average:     | {top5_with:16} | {top5_without}")
    print(f"Top 20 average:    | {top20_with:16} | {top20_without}")
    print(f"Top 100 average:   | {top100_with:16} | {top100_without}")
    
    print("\n----- Top 1 Score Statistics -----")
    top1_match_avg = f"{stats['avg_top1_score_match']:.5f}" if isinstance(stats['avg_top1_score_match'], float) else stats['avg_top1_score_match']
    top1_no_match_avg = f"{stats['avg_top1_score_no_match']:.5f}" if isinstance(stats['avg_top1_score_no_match'], float) else stats['avg_top1_score_no_match']
    
    print(f"Average score when top 1 is a match:     {top1_match_avg}")
    print(f"Average score when top 1 is not a match: {top1_no_match_avg}")
    print(f"Top 1 scores - matches count:            {len(stats['top1_scores_match'])}")
    print(f"Top 1 scores - non-matches count:        {len(stats['top1_scores_no_match'])}")
    
    print("\n----- Match Ranking Statistics -----")
    if isinstance(stats['avg_match_rank'], float):
        print(f"Average rank of first match: {stats['avg_match_rank']:.2f}")
        print(f"Median rank of first match: {stats['median_match_rank']:.2f}")
    else:
        print(f"Average rank of first match: {stats['avg_match_rank']}")
        print(f"Median rank of first match: {stats['median_match_rank']}")
    
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
    success_text += f"Top 1: {stats['top1_success_rate']:.2f}%\n"
    success_text += f"Top 5: {stats['top5_success_rate']:.2f}%\n"
    success_text += f"Top 20: {stats['top20_success_rate']:.2f}%\n"
    success_text += f"Top 100: {stats['top100_success_rate']:.2f}%\n"
    
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

def plot_top1_score_distribution_split(stats, output_file=None):
    """
    Create a histogram of top 1 scores, with separate colors for matches and non-matches
    """
    if not stats:
        print("No statistics available for plotting.")
        return
    
    if not stats['top1_scores_match'] and not stats['top1_scores_no_match']:
        print("No top 1 score data available for plotting.")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Get top 1 scores separated by match status
    top1_scores_match = np.array(stats['top1_scores_match'])
    top1_scores_no_match = np.array(stats['top1_scores_no_match'])
    
    # Get combined min and max for consistent binning
    all_scores = np.concatenate([top1_scores_match, top1_scores_no_match]) if len(top1_scores_match) > 0 and len(top1_scores_no_match) > 0 else \
                top1_scores_match if len(top1_scores_match) > 0 else top1_scores_no_match
                
    min_score = np.min(all_scores)
    max_score = np.max(all_scores)
    
    # Create bins
    bins = np.linspace(min_score, max_score, 30)
    
    # Plot histograms with different colors
    if len(top1_scores_match) > 0:
        plt.hist(top1_scores_match, bins=bins, alpha=0.7, color='green', 
                 label=f'Matches ({len(top1_scores_match)})', density=True)
    
    if len(top1_scores_no_match) > 0:
        plt.hist(top1_scores_no_match, bins=bins, alpha=0.7, color='red', 
                 label=f'Non-matches ({len(top1_scores_no_match)})', density=True)
    
    # Add labels and title
    plt.xlabel('Top 1 Similarity Score')
    plt.ylabel('Density')
    plt.title('Distribution of Top 1 Similarity Scores\n(Matches vs Non-matches)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Add statistics annotation
    match_avg = stats['avg_top1_score_match'] if isinstance(stats['avg_top1_score_match'], float) else 'N/A'
    no_match_avg = stats['avg_top1_score_no_match'] if isinstance(stats['avg_top1_score_no_match'], float) else 'N/A'
    
    stats_text = f"Top 1 Success Rate: {stats['top1_success_rate']:.2f}%\n"
    stats_text += f"Avg score (matches): {match_avg if isinstance(match_avg, str) else match_avg:.5f}\n"
    stats_text += f"Avg score (non-matches): {no_match_avg if isinstance(no_match_avg, str) else no_match_avg:.5f}"
    
    plt.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction',
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

def plot_top5_score_distribution_split(stats, output_file=None):
    """
    Create a histogram of top 5 average scores, with separate colors for matches and non-matches
    """
    if not stats:
        print("No statistics available for plotting.")
        return
    
    if not stats['top_5_match_in_top5'] and not stats['top_5_match_not_in_top5']:
        print("No top 5 score data available for plotting.")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Get top 5 scores separated by match status
    top5_scores_match = np.array(stats['top_5_match_in_top5'])
    top5_scores_no_match = np.array(stats['top_5_match_not_in_top5'])
    
    # Get combined min and max for consistent binning
    all_scores = np.concatenate([top5_scores_match, top5_scores_no_match]) if len(top5_scores_match) > 0 and len(top5_scores_no_match) > 0 else \
                top5_scores_match if len(top5_scores_match) > 0 else top5_scores_no_match
                
    min_score = np.min(all_scores)
    max_score = np.max(all_scores)
    
    # Create bins
    bins = np.linspace(min_score, max_score, 30)
    
    # Plot histograms with different colors
    if len(top5_scores_match) > 0:
        plt.hist(top5_scores_match, bins=bins, alpha=0.7, color='green', 
                 label=f'Match in top 5 ({len(top5_scores_match)})', density=True)
    
    if len(top5_scores_no_match) > 0:
        plt.hist(top5_scores_no_match, bins=bins, alpha=0.7, color='red', 
                 label=f'No match in top 5 ({len(top5_scores_no_match)})', density=True)
    
    # Add labels and title
    plt.xlabel('Average Similarity Score (Top 5)')
    plt.ylabel('Density')
    plt.title('Distribution of Top 5 Average Similarity Scores\n(Match in Top 5 vs No Match in Top 5)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Add statistics annotation
    match_avg = stats['avg_top_5_match_in_top5'] if isinstance(stats['avg_top_5_match_in_top5'], float) else 'N/A'
    no_match_avg = stats['avg_top_5_match_not_in_top5'] if isinstance(stats['avg_top_5_match_not_in_top5'], float) else 'N/A'
    
    stats_text = f"Top 5 Success Rate: {stats['top5_success_rate']:.2f}%\n"
    stats_text += f"Avg score (match in top 5): {match_avg if isinstance(match_avg, str) else match_avg:.5f}\n"
    stats_text += f"Avg score (no match in top 5): {no_match_avg if isinstance(no_match_avg, str) else no_match_avg:.5f}"
    
    plt.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction',
                 va='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save or display the plot
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_file, dpi=300)
        print(f"Top 5 score distribution plot saved to {output_file}")
    else:
        plt.show()

def plot_top20_score_distribution_split(stats, output_file=None):
    """
    Create a histogram of top 20 average scores, with separate colors for matches and non-matches
    """
    if not stats:
        print("No statistics available for plotting.")
        return
    
    if not stats['top_20_match_in_top20'] and not stats['top_20_match_not_in_top20']:
        print("No top 20 score data available for plotting.")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Get top 20 scores separated by match status
    top20_scores_match = np.array(stats['top_20_match_in_top20'])
    top20_scores_no_match = np.array(stats['top_20_match_not_in_top20'])
    
    # Get combined min and max for consistent binning
    all_scores = np.concatenate([top20_scores_match, top20_scores_no_match]) if len(top20_scores_match) > 0 and len(top20_scores_no_match) > 0 else \
                top20_scores_match if len(top20_scores_match) > 0 else top20_scores_no_match
                
    min_score = np.min(all_scores)
    max_score = np.max(all_scores)
    
    # Create bins
    bins = np.linspace(min_score, max_score, 30)
    
    # Plot histograms with different colors
    if len(top20_scores_match) > 0:
        plt.hist(top20_scores_match, bins=bins, alpha=0.7, color='green', 
                 label=f'Match in top 20 ({len(top20_scores_match)})', density=True)
    
    if len(top20_scores_no_match) > 0:
        plt.hist(top20_scores_no_match, bins=bins, alpha=0.7, color='red', 
                 label=f'No match in top 20 ({len(top20_scores_no_match)})', density=True)
    
    # Add labels and title
    plt.xlabel('Average Similarity Score (Top 20)')
    plt.ylabel('Density')
    plt.title('Distribution of Top 20 Average Similarity Scores\n(Match in Top 20 vs No Match in Top 20)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Add statistics annotation
    match_avg = stats['avg_top_20_match_in_top20'] if isinstance(stats['avg_top_20_match_in_top20'], float) else 'N/A'
    no_match_avg = stats['avg_top_20_match_not_in_top20'] if isinstance(stats['avg_top_20_match_not_in_top20'], float) else 'N/A'
    
    stats_text = f"Top 20 Success Rate: {stats['top20_success_rate']:.2f}%\n"
    stats_text += f"Avg score (match in top 20): {match_avg if isinstance(match_avg, str) else match_avg:.5f}\n"
    stats_text += f"Avg score (no match in top 20): {no_match_avg if isinstance(no_match_avg, str) else no_match_avg:.5f}"
    
    plt.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction',
                 va='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save or display the plot
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_file, dpi=300)
        print(f"Top 20 score distribution plot saved to {output_file}")
    else:
        plt.show()

def analyze_threshold_impact(file_pattern, match_threshold=0.3, score_threshold=0.6):
    """
    Analyze how applying a minimum similarity score threshold affects match probabilities.
    
    Args:
        file_pattern: Glob pattern to match CSV files or list of file paths
        match_threshold: Distance threshold for considering a match (in km)
        score_threshold: Minimum similarity score to consider a result
    
    Returns:
        Dictionary with comparison statistics
    """
    # Handle both string patterns and lists of files
    if isinstance(file_pattern, list):
        files = file_pattern
    else:
        files = glob.glob(file_pattern)
        
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return None
    
    print(f"Processing {len(files)} files to analyze impact of {score_threshold} score threshold")
    
    # Statistics storage
    stats = {
        # Base statistics
        'total_files': 0,
        'files_with_matches': 0,
        
        # Without threshold
        'top1_success_no_threshold': 0,
        'top5_success_no_threshold': 0,
        'top20_success_no_threshold': 0,
        'top100_success_no_threshold': 0,
        
        # Files that pass the threshold filter
        'files_with_top1_above_threshold': 0,
        'files_with_top1_above_threshold_and_matches': 0,
        'files_with_top1_above_threshold_success': 0,
        
        # Top 5 with threshold
        'files_with_top1_above_threshold_and_match_in_top5': 0,
        
        # Top 20 with threshold
        'files_with_top1_above_threshold_and_match_in_top20': 0,
        
        # Detailed tracking for histogram
        'match_ranks_no_threshold': [],
        'match_ranks_with_threshold': [],
        'top1_scores': [],
        'top1_scores_match': [],
        'top1_scores_no_match': []
    }
    
    # Process each file
    for file in files:
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
            
            # Track total files
            stats['total_files'] += 1
            
            # Get top 1 score and check if it's a match
            if len(df) > 0:
                top1_score = df.iloc[0]['similarity']
                top1_is_match = df.iloc[0]['is_match']
                
                # Store top 1 score
                stats['top1_scores'].append(top1_score)
                
                # Store based on match status
                if top1_is_match:
                    stats['top1_scores_match'].append(top1_score)
                else:
                    stats['top1_scores_no_match'].append(top1_score)
                
                # Check if top1 score is above threshold
                if top1_score >= score_threshold:
                    stats['files_with_top1_above_threshold'] += 1
                    
                    # Check if this is a successful match
                    if top1_is_match:
                        stats['files_with_top1_above_threshold_success'] += 1
            
            # Find if there are any matches in the dataset
            has_match = df['is_match'].any()
            
            if has_match:
                stats['files_with_matches'] += 1
                
                # Calculate statistics WITHOUT threshold
                
                # Find the rank of the first match (without threshold)
                first_match_rank_no_threshold = df[df['is_match']].index.min() + 1  # +1 because index is 0-based
                stats['match_ranks_no_threshold'].append(first_match_rank_no_threshold)
                
                # Check if match is in top N (without threshold)
                if first_match_rank_no_threshold <= 1:
                    stats['top1_success_no_threshold'] += 1
                if first_match_rank_no_threshold <= 5:
                    stats['top5_success_no_threshold'] += 1
                if first_match_rank_no_threshold <= 20:
                    stats['top20_success_no_threshold'] += 1
                if first_match_rank_no_threshold <= 100:
                    stats['top100_success_no_threshold'] += 1
                
                # Apply threshold and analyze impact on top 5 and top 20
                if len(df) > 0 and df.iloc[0]['similarity'] >= score_threshold:
                    # Count files with both matches and passing the threshold
                    stats['files_with_top1_above_threshold_and_matches'] += 1
                    
                    # Apply threshold filter
                    df_threshold = df[df['similarity'] >= score_threshold].copy()
                    
                    # Reset index after filtering
                    df_threshold.reset_index(drop=True, inplace=True)
                    
                    # Find rank of first match after threshold
                    if df_threshold['is_match'].any():
                        first_match_rank_with_threshold = df_threshold[df_threshold['is_match']].index.min() + 1
                        stats['match_ranks_with_threshold'].append(first_match_rank_with_threshold)
                        
                        # Check if match is in top 5 and top 20 after threshold
                        if first_match_rank_with_threshold <= 5:
                            stats['files_with_top1_above_threshold_and_match_in_top5'] += 1
                        if first_match_rank_with_threshold <= 20:
                            stats['files_with_top1_above_threshold_and_match_in_top20'] += 1
            
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}. Skipping.")
    
    print(f"Successfully processed {stats['total_files']} files")
    
    # Calculate probabilities
    if stats['total_files'] > 0:
        # Original probabilities (without threshold)
        stats['prob_top1_no_threshold'] = (stats['top1_success_no_threshold'] / stats['total_files']) * 100
        stats['prob_top5_no_threshold'] = (stats['top5_success_no_threshold'] / stats['total_files']) * 100
        stats['prob_top20_no_threshold'] = (stats['top20_success_no_threshold'] / stats['total_files']) * 100
        stats['prob_top100_no_threshold'] = (stats['top100_success_no_threshold'] / stats['total_files']) * 100
        
        # Probability that a file has a top1 score above threshold
        stats['prob_top1_above_threshold'] = (stats['files_with_top1_above_threshold'] / stats['total_files']) * 100
        
        # CONDITIONAL PROBABILITIES given score above threshold
        if stats['files_with_top1_above_threshold'] > 0:
            # Top 1 conditional probability
            stats['cond_prob_match_in_top1'] = (stats['files_with_top1_above_threshold_success'] / stats['files_with_top1_above_threshold']) * 100
            
            # Top 5 conditional probability
            stats['cond_prob_match_in_top5'] = (stats['files_with_top1_above_threshold_and_match_in_top5'] / stats['files_with_top1_above_threshold']) * 100
            
            # Top 20 conditional probability
            stats['cond_prob_match_in_top20'] = (stats['files_with_top1_above_threshold_and_match_in_top20'] / stats['files_with_top1_above_threshold']) * 100
        else:
            stats['cond_prob_match_in_top1'] = 0.0
            stats['cond_prob_match_in_top5'] = 0.0
            stats['cond_prob_match_in_top20'] = 0.0
    
    return stats

def print_threshold_analysis(stats, match_threshold, score_threshold):
    """
    Print the results of threshold impact analysis with corrected conditional probabilities
    """
    if not stats:
        print("No statistics available.")
        return
    
    print("\n====== THRESHOLD IMPACT ANALYSIS ======")
    print(f"Match threshold: {match_threshold} km")
    print(f"Score threshold: {score_threshold}\n")
    
    print(f"Total files processed: {stats['total_files']}")
    print(f"Files with at least one match (<{match_threshold}km): {stats['files_with_matches']} ({stats['files_with_matches']/stats['total_files']*100:.2f}%)")
    
    # Files that pass the threshold
    print(f"\nFiles with top 1 score ≥ {score_threshold}: {stats['files_with_top1_above_threshold']} ({stats['prob_top1_above_threshold']:.2f}%)")
    
    print("\n----- CONDITIONAL PROBABILITY ANALYSIS -----")
    print("Without using score threshold:")
    print(f"Probability of match in top 1: {stats['prob_top1_no_threshold']:.2f}%")
    print(f"Probability of match in top 5: {stats['prob_top5_no_threshold']:.2f}%")
    print(f"Probability of match in top 20: {stats['prob_top20_no_threshold']:.2f}%")
    
    print(f"\nWhen using score threshold ≥ {score_threshold}:")
    print(f"CONDITIONAL probability of match in top 1 given score ≥ {score_threshold}: {stats['cond_prob_match_in_top1']:.2f}%")
    print(f"CONDITIONAL probability of match in top 5 given score ≥ {score_threshold}: {stats['cond_prob_match_in_top5']:.2f}%")
    print(f"CONDITIONAL probability of match in top 20 given score ≥ {score_threshold}: {stats['cond_prob_match_in_top20']:.2f}%")
    
    # Change in probabilities
    change_top1 = stats['cond_prob_match_in_top1'] - stats['prob_top1_no_threshold']
    change_top5 = stats['cond_prob_match_in_top5'] - stats['prob_top5_no_threshold']
    change_top20 = stats['cond_prob_match_in_top20'] - stats['prob_top20_no_threshold']
    
    print(f"\nChange in probabilities:")
    print(f"Top 1: {change_top1:+.2f}%")
    print(f"Top 5: {change_top5:+.2f}%")
    print(f"Top 20: {change_top20:+.2f}%")
    
    # Detailed statistics about files retained vs. filtered
    print("\n----- THRESHOLD FILTERING STATISTICS -----")
    print(f"Files retained after filtering (top 1 score ≥ {score_threshold}): {stats['files_with_top1_above_threshold']} of {stats['total_files']} ({stats['files_with_top1_above_threshold']/stats['total_files']*100:.2f}%)")
    print(f"Files with matches retained after filtering: {stats['files_with_top1_above_threshold_and_matches']} of {stats['files_with_matches']} ({stats['files_with_top1_above_threshold_and_matches']/stats['files_with_matches']*100:.2f}% of files with matches)")
    
    print("=========================================")

def plot_top1_score_threshold_impact(stats, score_threshold, output_file=None):
    """
    Create a plot showing the impact of top1 score threshold on match probability
    """
    if not stats:
        print("No statistics available for plotting.")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot top 1 scores distribution, separated by match vs non-match
    top1_scores_match = np.array(stats['top1_scores_match'])
    top1_scores_no_match = np.array(stats['top1_scores_no_match'])
    
    # Get combined min and max for consistent binning
    all_scores = np.array(stats['top1_scores'])
    min_score = np.min(all_scores)
    max_score = np.max(all_scores)
    
    # Create bins
    bins = np.linspace(min_score, max_score, 30)
    
    # Plot histograms
    if len(top1_scores_match) > 0:
        plt.hist(top1_scores_match, bins=bins, alpha=0.7, color='green', 
                 label=f'Matches ({len(top1_scores_match)})', density=True)
    
    if len(top1_scores_no_match) > 0:
        plt.hist(top1_scores_no_match, bins=bins, alpha=0.7, color='red', 
                 label=f'Non-matches ({len(top1_scores_no_match)})', density=True)
    
    # Add a vertical line at the threshold
    plt.axvline(x=score_threshold, color='black', linestyle='--', linewidth=2, 
                label=f'Threshold: {score_threshold}')
    
    # Add statistics annotation
    total_above = stats['files_with_top1_above_threshold']
    total_files = stats['total_files']
    match_rate_all = stats['prob_top1_no_threshold']
    match_rate_above = stats['cond_prob_match_in_top1']
    
    stats_text = f"Files with top 1 score ≥ {score_threshold}: {total_above}/{total_files} ({total_above/total_files*100:.1f}%)\n"
    stats_text += f"Match rate without threshold: {match_rate_all:.1f}%\n"
    stats_text += f"Match rate with threshold: {match_rate_above:.1f}%\n"
    stats_text += f"Change: {match_rate_above - match_rate_all:+.1f}%"
    
    plt.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction',
                 va='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                 fontsize=11)
    
    # Add labels and title
    plt.xlabel('Top 1 Similarity Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Distribution of Top 1 Scores and Impact of {score_threshold} Threshold', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save or display the plot
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_file, dpi=300)
        print(f"Threshold impact plot saved to {output_file}")
    else:
        plt.show()

def plot_threshold_comparison_bar(stats, score_threshold, output_file=None):
    """
    Create a bar chart comparing success rates with and without threshold
    """
    if not stats:
        print("No statistics available for plotting.")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Data
    categories = ['Top 1', 'Top 5', 'Top 20']
    no_threshold = [
        stats['prob_top1_no_threshold'],
        stats['prob_top5_no_threshold'],
        stats['prob_top20_no_threshold']
    ]
    with_threshold = [
        stats['cond_prob_match_in_top1'],
        stats['cond_prob_match_in_top5'],
        stats['cond_prob_match_in_top20']
    ]
    
    # Calculate differences for coloring
    differences = [with_threshold[i] - no_threshold[i] for i in range(len(categories))]
    
    # Bar positions
    x = np.arange(len(categories))
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, no_threshold, width, label=f'No threshold', color='blue', alpha=0.7)
    
    # Use colors based on improvement
    bar_colors = ['green' if diff > 0 else 'red' for diff in differences]
    bars = plt.bar(x + width/2, with_threshold, width, label=f'Score threshold ≥ {score_threshold}', color=bar_colors, alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(no_threshold):
        plt.text(i - width/2, v + 1, f"{v:.1f}%", ha='center', fontsize=11)
    
    for i, v in enumerate(with_threshold):
        plt.text(i + width/2, v + 1, f"{v:.1f}%", ha='center', fontsize=11)
        
        # Add the difference
        diff = differences[i]
        color = 'green' if diff > 0 else 'red'
        plt.text(i + width/2, v + 5, f"{diff:+.1f}%", ha='center', fontsize=10, color=color, fontweight='bold')
    
    # Add labels and title
    plt.xlabel('Rank Cutoff', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title(f'Match Success Rate Comparison\nWith and Without {score_threshold} Similarity Score Threshold', fontsize=14)
    plt.xticks(x, categories, fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    
    # Add annotations with key statistics
    total_above = stats['files_with_top1_above_threshold']
    total_files = stats['total_files']
    
    stats_text = f"Files with top 1 score ≥ {score_threshold}: {total_above}/{total_files} ({total_above/total_files*100:.1f}%)\n\n"
    stats_text += "Impact of threshold filter:\n"
    for i, cat in enumerate(categories):
        diff = differences[i]
        stats_text += f"{cat}: {diff:+.1f}% {'improvement' if diff > 0 else 'decrease'}\n"
    
    plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction',
                 va='bottom', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                 fontsize=10)
    
    plt.tight_layout()
    
    # Save or display the plot
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_file, dpi=300)
        print(f"Threshold comparison bar chart saved to {output_file}")
    else:
        plt.show()

def plot_combined_top_n_distributions(stats, output_file=None):
    """
    Create a single figure with three subplots showing the distributions of top 1, top 5, and top 20 scores side by side
    """
    if not stats:
        print("No statistics available for plotting.")
        return
    
    # Check if we have data for the plots
    has_top1_data = bool(stats['top1_scores_match'] or stats['top1_scores_no_match'])
    has_top5_data = bool(stats['top_5_match_in_top5'] or stats['top_5_match_not_in_top5'])
    has_top20_data = bool(stats['top_20_match_in_top20'] or stats['top_20_match_not_in_top20'])
    
    print(f"Debug - Data availability: Top1={has_top1_data}, Top5={has_top5_data}, Top20={has_top20_data}")
    
    if not (has_top1_data or has_top5_data or has_top20_data):
        print("No score data available for plotting.")
        return
    
    # Create a figure with three subplots side by side
    print("Creating figure with 3 subplots side by side...")
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    
    # Plot Top 1 distribution
    if has_top1_data:
        print("Plotting Top 1 distribution...")
        ax = axs[0]
        top1_scores_match = np.array(stats['top1_scores_match'])
        top1_scores_no_match = np.array(stats['top1_scores_no_match'])
        
        print(f"  Top1 data counts: matches={len(top1_scores_match)}, non-matches={len(top1_scores_no_match)}")
        
        # Get combined min and max for consistent binning
        all_scores = np.concatenate([top1_scores_match, top1_scores_no_match]) if len(top1_scores_match) > 0 and len(top1_scores_no_match) > 0 else \
                    top1_scores_match if len(top1_scores_match) > 0 else top1_scores_no_match
        
        min_score = np.min(all_scores)
        max_score = np.max(all_scores)
        
        # Create bins
        bins = np.linspace(min_score, max_score, 30)
        
        # Plot histograms with different colors
        if len(top1_scores_match) > 0:
            ax.hist(top1_scores_match, bins=bins, alpha=0.7, color='green', 
                    label=f'Matches ({len(top1_scores_match)})', density=True)
        
        if len(top1_scores_no_match) > 0:
            ax.hist(top1_scores_no_match, bins=bins, alpha=0.7, color='red', 
                    label=f'Non-matches ({len(top1_scores_no_match)})', density=True)
        
        # Add labels and title
        ax.set_xlabel('Top 1 Similarity Score')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Top 1 Similarity Scores\n(Matches vs Non-matches)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Add statistics annotation
        match_avg = stats['avg_top1_score_match'] if isinstance(stats['avg_top1_score_match'], float) else 'N/A'
        no_match_avg = stats['avg_top1_score_no_match'] if isinstance(stats['avg_top1_score_no_match'], float) else 'N/A'
        
        stats_text = f"Top 1 Success Rate: {stats['top1_success_rate']:.2f}%\n"
        stats_text += f"Avg score (matches): {match_avg if isinstance(match_avg, str) else match_avg:.5f}\n"
        stats_text += f"Avg score (non-matches): {no_match_avg if isinstance(no_match_avg, str) else no_match_avg:.5f}"
        
        ax.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction',
                    va='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    else:
        axs[0].text(0.5, 0.5, 'No Top 1 data available', 
                    horizontalalignment='center', verticalalignment='center')
    
    # Plot Top 5 distribution
    if has_top5_data:
        print("Plotting Top 5 distribution...")
        ax = axs[1]
        top5_scores_match = np.array(stats['top_5_match_in_top5'])
        top5_scores_no_match = np.array(stats['top_5_match_not_in_top5'])
        
        print(f"  Top5 data counts: matches={len(top5_scores_match)}, non-matches={len(top5_scores_no_match)}")
        
        # Get combined min and max for consistent binning
        all_scores = np.concatenate([top5_scores_match, top5_scores_no_match]) if len(top5_scores_match) > 0 and len(top5_scores_no_match) > 0 else \
                    top5_scores_match if len(top5_scores_match) > 0 else top5_scores_no_match
                    
        min_score = np.min(all_scores)
        max_score = np.max(all_scores)
        
        # Create bins
        bins = np.linspace(min_score, max_score, 30)
        
        # Plot histograms with different colors
        if len(top5_scores_match) > 0:
            ax.hist(top5_scores_match, bins=bins, alpha=0.7, color='green', 
                    label=f'Match in top 5 ({len(top5_scores_match)})', density=True)
        
        if len(top5_scores_no_match) > 0:
            ax.hist(top5_scores_no_match, bins=bins, alpha=0.7, color='red', 
                    label=f'No match in top 5 ({len(top5_scores_no_match)})', density=True)
        
        # Add labels and title
        ax.set_xlabel('Average Similarity Score (Top 5)')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Top 5 Average Similarity Scores\n(Match in Top 5 vs No Match in Top 5)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Add statistics annotation
        match_avg = stats['avg_top_5_match_in_top5'] if isinstance(stats['avg_top_5_match_in_top5'], float) else 'N/A'
        no_match_avg = stats['avg_top_5_match_not_in_top5'] if isinstance(stats['avg_top_5_match_not_in_top5'], float) else 'N/A'
        
        stats_text = f"Top 5 Success Rate: {stats['top5_success_rate']:.2f}%\n"
        stats_text += f"Avg score (match in top 5): {match_avg if isinstance(match_avg, str) else match_avg:.5f}\n"
        stats_text += f"Avg score (no match in top 5): {no_match_avg if isinstance(no_match_avg, str) else no_match_avg:.5f}"
        
        ax.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction',
                    va='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    else:
        axs[1].text(0.5, 0.5, 'No Top 5 data available', 
                    horizontalalignment='center', verticalalignment='center')
    
    # Plot Top 20 distribution
    if has_top20_data:
        print("Plotting Top 20 distribution...")
        ax = axs[2]
        top20_scores_match = np.array(stats['top_20_match_in_top20'])
        top20_scores_no_match = np.array(stats['top_20_match_not_in_top20'])
        
        print(f"  Top20 data counts: matches={len(top20_scores_match)}, non-matches={len(top20_scores_no_match)}")
        
        # Get combined min and max for consistent binning
        all_scores = np.concatenate([top20_scores_match, top20_scores_no_match]) if len(top20_scores_match) > 0 and len(top20_scores_no_match) > 0 else \
                    top20_scores_match if len(top20_scores_match) > 0 else top20_scores_no_match
                    
        min_score = np.min(all_scores)
        max_score = np.max(all_scores)
        
        # Create bins
        bins = np.linspace(min_score, max_score, 30)
        
        # Plot histograms with different colors
        if len(top20_scores_match) > 0:
            ax.hist(top20_scores_match, bins=bins, alpha=0.7, color='green', 
                    label=f'Match in top 20 ({len(top20_scores_match)})', density=True)
        
        if len(top20_scores_no_match) > 0:
            ax.hist(top20_scores_no_match, bins=bins, alpha=0.7, color='red', 
                    label=f'No match in top 20 ({len(top20_scores_no_match)})', density=True)
        
        # Add labels and title
        ax.set_xlabel('Average Similarity Score (Top 20)')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Top 20 Average Similarity Scores\n(Match in Top 20 vs No Match in Top 20)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Add statistics annotation
        match_avg = stats['avg_top_20_match_in_top20'] if isinstance(stats['avg_top_20_match_in_top20'], float) else 'N/A'
        no_match_avg = stats['avg_top_20_match_not_in_top20'] if isinstance(stats['avg_top_20_match_not_in_top20'], float) else 'N/A'
        
        stats_text = f"Top 20 Success Rate: {stats['top20_success_rate']:.2f}%\n"
        stats_text += f"Avg score (match in top 20): {match_avg if isinstance(match_avg, str) else match_avg:.5f}\n"
        stats_text += f"Avg score (no match in top 20): {no_match_avg if isinstance(no_match_avg, str) else no_match_avg:.5f}"
        
        ax.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction',
                    va='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    else:
        axs[2].text(0.5, 0.5, 'No Top 20 data available', 
                    horizontalalignment='center', verticalalignment='center')
    
    # Add a shared title for the entire figure
    fig.suptitle(f'Top N Similarity Score Distributions', fontsize=16)
    
    # Save or display the plot
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        print(f"Saving combined plot to {output_file}...")
        plt.savefig(output_file, dpi=300)
        print(f"Combined distributions plot saved to {output_file}")
    else:
        print("Displaying combined plot...")
        plt.show()
    
    # Explicitly close the figure to ensure it's displayed/saved properly
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ranking statistics from image retrieval results with focus on top N cutoffs")
    parser.add_argument("file_pattern", help="Glob pattern for CSV files (e.g., '/path/to/files/*.csv')")
    parser.add_argument("--threshold", type=float, default=0.6, help="Distance threshold for matches (km)")
    parser.add_argument("--output", help="Output directory for plots", default=None)
    parser.add_argument("--exclude", help="Pattern to exclude certain files", default=None)
    parser.add_argument("--score_threshold", type=float, default=None, help="Analyze impact of minimum similarity score threshold")
    parser.add_argument("--combined", action="store_true", help="Create a combined plot with all distributions")
    
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
    
    # Perform score threshold analysis if requested
    if args.score_threshold is not None:
        # Analyze impact of applying a score threshold
        threshold_stats = analyze_threshold_impact(args.file_pattern, args.threshold, args.score_threshold)
        
        if threshold_stats:
            # Print detailed statistics
            print_threshold_analysis(threshold_stats, args.threshold, args.score_threshold)
            
            # Generate plots
            if args.output:
                # Create the output directory if it doesn't exist
                os.makedirs(args.output, exist_ok=True)
                
                # Histogram with threshold line
                threshold_plot_file = os.path.join(args.output, f"threshold_{args.score_threshold}_impact.png")
                plot_top1_score_threshold_impact(threshold_stats, args.score_threshold, threshold_plot_file)
                
                # Bar chart comparison
                bar_chart_file = os.path.join(args.output, f"threshold_{args.score_threshold}_comparison_bar.png")
                plot_threshold_comparison_bar(threshold_stats, args.score_threshold, bar_chart_file)
            else:
                plot_top1_score_threshold_impact(threshold_stats, args.score_threshold)
                plot_threshold_comparison_bar(threshold_stats, args.score_threshold)
    else:
        # Regular analysis
        stats = analyze_rankings_by_top_n(args.file_pattern, args.threshold)
        
        if stats:
            # Print detailed statistics
            print_statistics_by_top_n(stats, args.threshold)
            
            # Plot match rank distribution if we have match data
            if stats['files_with_matches'] > 0:
                
                if args.combined:
                    print("Creating combined plot with Top 1, Top 5, and Top 20 distributions...")
                    # Create a combined plot with all distributions
                    if args.output:
                        os.makedirs(args.output, exist_ok=True)
                        combined_plot_file = os.path.join(args.output, "combined_top_n_distributions.png") 
                        plot_combined_top_n_distributions(stats, combined_plot_file)
                    else:
                        plot_combined_top_n_distributions(stats)
                        
                    # Always display the match rank distribution plot
                    if args.output:
                        rank_plot_file = os.path.join(args.output, "match_rank_distribution_by_top_n.png")
                        plot_match_ranks(stats, rank_plot_file)
                    else:
                        plot_match_ranks(stats)
                else:
                    # Display or save all plots individually
                    if args.output:
                        # Create the output directory if it doesn't exist
                        os.makedirs(args.output, exist_ok=True)
                        
                        # Save the match rank distribution plot
                        rank_plot_file = os.path.join(args.output, "match_rank_distribution_by_top_n.png")
                        plot_match_ranks(stats, rank_plot_file)
                        
                        # Save separate plots
                        print("Creating separate plots for Top 1, Top 5, and Top 20 distributions...")
                        top1_plot_file = os.path.join(args.output, "top1_score_distribution_split.png")
                        plot_top1_score_distribution_split(stats, top1_plot_file)
                        
                        top5_plot_file = os.path.join(args.output, "top5_score_distribution_split.png")
                        plot_top5_score_distribution_split(stats, top5_plot_file)
                        
                        top20_plot_file = os.path.join(args.output, "top20_score_distribution_split.png")
                        plot_top20_score_distribution_split(stats, top20_plot_file)
                    else:
                        # Just show the plots
                        plot_match_ranks(stats)
                        plot_top1_score_distribution_split(stats)
                        plt.figure()  # Create a new figure for the next plot
                        plot_top5_score_distribution_split(stats)
                        plt.figure()  # Create a new figure for the next plot
                        plot_top20_score_distribution_split(stats)
            else:
                print("No match data available for plotting.")
        else:
            print("No data available for analysis.") 
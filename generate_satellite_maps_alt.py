import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import argparse
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects
from matplotlib.colors import LinearSegmentedColormap
import smopy
from PIL import Image
import requests
from io import BytesIO

def load_csv_files(file_pattern):
    """
    Load all CSV files matching the pattern and combine them into a single DataFrame.
    
    Args:
        file_pattern: Glob pattern for CSV files (e.g., 'tilt_corrected_*_results.csv')
    
    Returns:
        Combined DataFrame with data from all files
    """
    files = glob.glob(file_pattern)
    if not files:
        raise ValueError(f"No files found matching pattern: {file_pattern}")
    
    print(f"Found {len(files)} files matching pattern")
    
    all_data = []
    
    for file in files:
        print(f"Processing {file}...")
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Skip the first row which is the reference image
        df = df.iloc[1:].copy()
        
        # Convert distance_km and similarity to numeric values
        df['distance_km'] = pd.to_numeric(df['distance_km'], errors='coerce')
        df['similarity'] = pd.to_numeric(df['similarity'], errors='coerce')
        
        # Add file name as source
        df['source_file'] = os.path.basename(file)
        
        # Add to the combined dataset
        all_data.append(df)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def get_satellite_background(min_lat, max_lat, min_lon, max_lon, zoom=17):
    """
    Get a high-resolution satellite background image using Smopy.
    
    Args:
        min_lat, max_lat, min_lon, max_lon: Bounding box coordinates
        zoom: Zoom level for the map (higher = more detailed, default increased to 17)
    
    Returns:
        Map image and extent information for displaying
    """
    try:
        # Try to get the highest possible zoom level
        # More zoomed = higher resolution
        # Create a map with a slightly larger area to ensure coverage
        padding = 0.01
        
        # First try with the requested zoom level
        try:
            print(f"Attempting to fetch satellite imagery at zoom level {zoom}...")
            map_area = smopy.Map(
                (min_lat-padding, min_lon-padding, max_lat+padding, max_lon+padding),
                z=zoom,
                tileserver="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            )
            print(f"Successfully retrieved satellite imagery at zoom level {zoom}")
        except Exception as zoom_error:
            # If that fails, try a lower zoom level
            fallback_zoom = 15
            print(f"Failed at zoom level {zoom}, falling back to zoom level {fallback_zoom}: {zoom_error}")
            map_area = smopy.Map(
                (min_lat-padding, min_lon-padding, max_lat+padding, max_lon+padding),
                z=fallback_zoom,
                tileserver="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            )
            print(f"Successfully retrieved satellite imagery at fallback zoom level {fallback_zoom}")
        
        # Calculate the proper extent for the image
        # Smopy uses y-origin at the top, but imshow uses y-origin at the bottom
        # So we need to flip the y-coordinates
        box = map_area.box
        extent = [box[1], box[3], box[2], box[0]]
        
        return map_area.img, extent
    except Exception as e:
        print(f"Warning: Could not load satellite imagery with Smopy: {e}")
        print("Using alternative method...")
        
        # Try direct download as backup method with larger image size
        try:
            url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
            # Request a higher resolution image (3600x3600 instead of 2400x2400)
            params = {
                'bbox': f"{min_lon},{min_lat},{max_lon},{max_lat}",
                'bboxSR': '4326',
                'imageSR': '4326',
                'size': '3600,3600',  # Higher resolution
                'format': 'png',
                'f': 'image'
            }
            
            print("Fetching high-resolution image via direct API call...")
            response = requests.get(url, params=params)
            img = Image.open(BytesIO(response.content))
            img_array = np.array(img)
            print(f"Successfully retrieved high-resolution image: {img.width}x{img.height} pixels")
            
            # The extent for this image is simply the bounding box coordinates
            extent = [min_lon, max_lon, min_lat, max_lat]
                
            return img_array, extent
        except Exception as e2:
            print(f"Warning: Alternative method also failed: {e2}")
            print("Using plain background.")
            return None, None

def create_satellite_maps(df, output_dir='satellite_maps_alt', match_threshold=0.3, dpi=1200):
    """
    Create image maps with satellite imagery as the background.
    
    Args:
        df: Combined DataFrame with all tile data
        output_dir: Directory to save the output image files
        match_threshold: Distance threshold for considering a match (in km)
        dpi: Resolution of output images (increased to 1200 for better zooming)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out rows with NaN or N/A coordinates
    valid_df = df[
        (df['latitude'] != 'N/A') & 
        (df['longitude'] != 'N/A') &
        pd.notna(df['latitude']) & 
        pd.notna(df['longitude'])
    ].copy()
    
    # Convert coordinates to numeric
    valid_df['latitude'] = pd.to_numeric(valid_df['latitude'])
    valid_df['longitude'] = pd.to_numeric(valid_df['longitude'])
    
    # Calculate frequency and average similarity for each tile
    tile_stats = valid_df.groupby(['latitude', 'longitude']).agg({
        'similarity': 'mean',
        'source_file': 'count'
    }).reset_index()
    
    tile_stats.columns = ['latitude', 'longitude', 'avg_similarity', 'frequency']
    
    # Identify match tiles
    match_df = valid_df[valid_df['distance_km'] < match_threshold].copy()
    match_df = match_df.drop_duplicates(['latitude', 'longitude'])
    
    # Prepare ground truth points (best match for each source)
    image_sources = df['source_file'].unique()
    ground_truth_points = []
    
    for source in image_sources:
        source_df = valid_df[valid_df['source_file'] == source]
        matches = source_df[source_df['distance_km'] < match_threshold].copy()
        
        if not matches.empty:
            # Take the best match (first in sorted order)
            best_match = matches.iloc[0]
            ground_truth_points.append({
                'latitude': best_match['latitude'],
                'longitude': best_match['longitude'],
                'source': source,
                'image_name': best_match['image_name'],
                'distance_km': best_match['distance_km'],
                'similarity': best_match['similarity']
            })
    
    # Create a DataFrame for ground truth points
    if ground_truth_points:
        gt_df = pd.DataFrame(ground_truth_points)
    else:
        gt_df = pd.DataFrame(columns=['latitude', 'longitude', 'source', 'image_name', 'distance_km', 'similarity'])
    
    # Set up common plot parameters
    plt.rcParams.update({'font.size': 12})
    
    # Calculate plot bounds to set consistent zoom level
    min_lon = valid_df['longitude'].min() - 0.05
    max_lon = valid_df['longitude'].max() + 0.05
    min_lat = valid_df['latitude'].min() - 0.05
    max_lat = valid_df['latitude'].max() + 0.05
    
    # Get high-resolution satellite background
    map_img, extent = get_satellite_background(min_lat, max_lat, min_lon, max_lon, zoom=15)
    
    # Create empty satellite map (just the background)
    if map_img is not None:
        fig, ax = plt.subplots(figsize=(16, 14))
        
        try:
            ax.imshow(map_img, extent=extent, origin='upper')
            
            # Set the x and y limits to our calculated bounds
            ax.set_xlim(min_lon, max_lon)
            ax.set_ylim(min_lat, max_lat)
            
            # Add grid lines with low opacity for reference
            ax.grid(alpha=0.15)
            
            # Add attribution
            ax.text(0.02, 0.02, f'Map data © Esri', transform=ax.transAxes, 
                    fontsize=8, color='white', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
            
            plt.tight_layout()
            
            # Save the empty satellite map
            empty_file = os.path.join(output_dir, 'empty_satellite_map.png')
            plt.savefig(empty_file, dpi=dpi, bbox_inches='tight')
            plt.close()
            print(f"Empty satellite map saved to {empty_file}")
        except Exception as e:
            print(f"Warning: Could not create empty satellite map: {e}")
    else:
        print("Warning: Could not create empty satellite map because satellite imagery could not be loaded")
    
    # Calculate values needed for plotting
    max_freq = tile_stats['frequency'].max()
    min_freq = tile_stats['frequency'].min()
    sizes_legend = [min_freq, (min_freq + max_freq) / 2, max_freq]
    legend_sizes = [30 + (size - min_freq) / (max_freq - min_freq) * 250 for size in sizes_legend]
    
    # Create custom colormap for better visibility on satellite imagery
    colors = ['blue', 'cyan', 'yellow', 'red']
    cmap = LinearSegmentedColormap.from_list("confidence", colors)
    
    # 1. Create frequency map with satellite background
    fig, ax = plt.subplots(figsize=(20, 18))  # Increased figure size for more detail
    
    # Add satellite background if available
    if map_img is not None:
        try:
            ax.imshow(map_img, extent=extent, origin='upper')
        except Exception as e:
            print(f"Warning: Could not display satellite background: {e}")
    
    # Set up the scatter plot with sizes based on frequency
    sizes = 30 + (tile_stats['frequency'] - min_freq) / (max_freq - min_freq) * 250
    
    # Create a more opaque blue color for better visibility on satellite imagery
    scatter = ax.scatter(
        tile_stats['longitude'], 
        tile_stats['latitude'],
        s=sizes,
        c='royalblue',
        edgecolor='white',
        linewidth=0.5,
        alpha=0.8,
        label='Tiles'
    )
    
    # Add match tiles as green markers with white edge for visibility
    if not match_df.empty:
        ax.scatter(
            match_df['longitude'], 
            match_df['latitude'],
            s=80,
            c='green',
            marker='o',
            edgecolor='white',
            linewidth=1,
            alpha=0.9,
            label=f'Matches (<{match_threshold} km)'
        )
    
    # Add title and labels
    title = ax.set_title('Frequency of Tiles Returned from UAV Image Retrieval', fontsize=16)
    title.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    
    # Set the x and y limits to our calculated bounds
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    
    # Add a size legend with white outline for visibility
    size_handles = [
        Line2D([0], [0], marker='o', color='royalblue', markerfacecolor='royalblue', 
               markeredgecolor='white', markeredgewidth=0.5,
               markersize=np.sqrt(size)/4, alpha=0.8, linestyle='')
        for size in legend_sizes
    ]
    
    # Create a second legend for sizes
    first_legend = ax.legend(loc='upper right', framealpha=0.9)
    for text in first_legend.get_texts():
        text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
        
    ax.add_artist(first_legend)
    second_legend = ax.legend(size_handles, [f"{int(size)}" for size in sizes_legend], 
               title="Frequency", loc='upper left', framealpha=0.9)
    for text in second_legend.get_texts():
        text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    second_legend.get_title().set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    
    # Add grid lines with low opacity
    ax.grid(alpha=0.15)
    
    # Add attribution
    ax.text(0.02, 0.02, f'Map data © Esri', transform=ax.transAxes, 
            fontsize=8, color='white', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    
    # Save the frequency map
    freq_file = os.path.join(output_dir, 'frequency_satellite_map.png')
    plt.savefig(freq_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Frequency satellite map saved to {freq_file}")
    
    # 2. Create confidence map with satellite background
    fig, ax = plt.subplots(figsize=(20, 18))  # Increased figure size for more detail
    
    # Add satellite background if available
    if map_img is not None:
        try:
            ax.imshow(map_img, extent=extent, origin='upper')
        except Exception as e:
            print(f"Warning: Could not display satellite background: {e}")
    
    # Set up the scatter plot with colors based on confidence
    norm = plt.Normalize(tile_stats['avg_similarity'].min(), tile_stats['avg_similarity'].max())
    
    scatter = ax.scatter(
        tile_stats['longitude'], 
        tile_stats['latitude'],
        s=80,
        c=tile_stats['avg_similarity'],
        cmap=cmap,
        edgecolor='white',
        linewidth=0.5,
        alpha=0.9,
        label='Tiles',
        norm=norm
    )
    
    # Add match tiles as green markers with white edge for visibility
    if not match_df.empty:
        ax.scatter(
            match_df['longitude'], 
            match_df['latitude'],
            s=80,
            c='lime',
            marker='o',
            edgecolor='white',
            linewidth=1,
            alpha=0.9,
            label=f'Matches (<{match_threshold} km)'
        )
    
    # Add title and labels
    title = ax.set_title('Average Confidence (Similarity) of Tiles', fontsize=16)
    title.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    
    # Set the x and y limits to our calculated bounds
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    
    # Add a colorbar with white outline
    cbar = fig.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label('Average Similarity Score', fontsize=14)
    cbar.ax.yaxis.label.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    for t in cbar.ax.get_yticklabels():
        t.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    
    # Make the legend have a semi-transparent background
    legend = ax.legend(loc='upper right', framealpha=0.9)
    for text in legend.get_texts():
        text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    
    # Add grid lines with low opacity
    ax.grid(alpha=0.15)
    
    # Add attribution
    ax.text(0.02, 0.02, f'Map data © Esri', transform=ax.transAxes, 
            fontsize=8, color='white', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    
    # Save the confidence map
    conf_file = os.path.join(output_dir, 'confidence_satellite_map.png')
    plt.savefig(conf_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Confidence satellite map saved to {conf_file}")
    
    # 3. Create ground truth map with satellite background
    fig, ax = plt.subplots(figsize=(20, 18))  # Increased figure size for more detail
    
    # Add satellite background if available
    if map_img is not None:
        try:
            ax.imshow(map_img, extent=extent, origin='upper')
        except Exception as e:
            print(f"Warning: Could not display satellite background: {e}")
    
    # Add a heatmap of all data points with transparency
    heatmap = ax.hexbin(
        valid_df['longitude'], 
        valid_df['latitude'],
        gridsize=40,
        cmap='Blues',
        alpha=0.3,  # 30% opacity
        mincnt=1
    )
    
    # Add ground truth points with star markers
    if not gt_df.empty:
        ax.scatter(
            gt_df['longitude'], 
            gt_df['latitude'],
            s=200,
            c='yellow',
            marker='*',
            edgecolor='black',
            linewidth=1.5,
            alpha=1.0,
            label='Ground Truth',
            zorder=10
        )
        
        # Annotate ground truth points with white outline for visibility
        for i, row in gt_df.iterrows():
            text = ax.annotate(
                f"{os.path.basename(row['source']).split('_')[2]}",
                (row['longitude'], row['latitude']),
                xytext=(10, 5),
                textcoords='offset points',
                fontsize=10,
                color='white',
                weight='bold',
                bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.2'),
                zorder=11
            )
    
    # Add title with white outline
    title = ax.set_title('Ground Truth Locations (Best Matches for Each UAV Image)', fontsize=16)
    title.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    
    # Set the x and y limits to our calculated bounds
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    
    # Add a colorbar for the heatmap
    cbar = fig.colorbar(heatmap, ax=ax, pad=0.01)
    cbar.set_label('Data Point Density', fontsize=14)
    cbar.ax.yaxis.label.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    for t in cbar.ax.get_yticklabels():
        t.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    
    # Make the legend have a semi-transparent background
    legend = ax.legend(loc='upper right', framealpha=0.9)
    for text in legend.get_texts():
        text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    
    # Add grid lines with low opacity
    ax.grid(alpha=0.15)
    
    # Add attribution
    ax.text(0.02, 0.02, f'Map data © Esri', transform=ax.transAxes, 
            fontsize=8, color='white', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    
    # Save the ground truth map
    gt_file = os.path.join(output_dir, 'ground_truth_satellite_map.png')
    plt.savefig(gt_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Ground truth satellite map saved to {gt_file}")
    
    # 4. Create combined map with satellite background
    fig, ax = plt.subplots(figsize=(20, 18))  # Increased figure size for more detail
    
    # Add satellite background if available
    if map_img is not None:
        try:
            ax.imshow(map_img, extent=extent, origin='upper')
        except Exception as e:
            print(f"Warning: Could not display satellite background: {e}")
    
    # Set up the scatter plot with sizes based on frequency and colors based on confidence
    norm = plt.Normalize(tile_stats['avg_similarity'].min(), tile_stats['avg_similarity'].max())
    
    # Use custom colormap for better visibility
    scatter = ax.scatter(
        tile_stats['longitude'], 
        tile_stats['latitude'],
        s=30 + (tile_stats['frequency'] - min_freq) / (max_freq - min_freq) * 250,
        c=tile_stats['avg_similarity'],
        cmap=cmap,
        edgecolor='white',
        linewidth=0.5,
        alpha=0.8,
        label='Tiles',
        norm=norm
    )
    
    # Add ground truth points
    if not gt_df.empty:
        ax.scatter(
            gt_df['longitude'], 
            gt_df['latitude'],
            s=200,
            c='yellow',
            marker='*',
            edgecolor='black',
            linewidth=1.5,
            alpha=1.0,
            label='Ground Truth',
            zorder=10
        )
    
    # Add title with white outline
    title = ax.set_title('Combined: Frequency (size) and Confidence (color)', fontsize=16)
    title.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    
    # Set the x and y limits to our calculated bounds
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    
    # Add a colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label('Similarity Score', fontsize=14)
    cbar.ax.yaxis.label.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    for t in cbar.ax.get_yticklabels():
        t.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    
    # Create a legend for sizes
    size_handles = [
        Line2D([0], [0], marker='o', color=cmap(0.7), markerfacecolor=cmap(0.7), 
               markeredgecolor='white', markeredgewidth=0.5,
               markersize=np.sqrt(size)/4, alpha=0.8, linestyle='')
        for size in legend_sizes
    ]
    
    # Create a legend with transparent background
    first_legend = ax.legend(loc='upper right', framealpha=0.9)
    for text in first_legend.get_texts():
        text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
        
    ax.add_artist(first_legend)
    second_legend = ax.legend(size_handles, [f"{int(size)}" for size in sizes_legend], 
               title="Frequency", loc='upper left', framealpha=0.9)
    for text in second_legend.get_texts():
        text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    second_legend.get_title().set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    
    # Add grid lines with low opacity
    ax.grid(alpha=0.15)
    
    # Add attribution
    ax.text(0.02, 0.02, f'Map data © Esri', transform=ax.transAxes, 
            fontsize=8, color='white', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
    
    plt.tight_layout()
    
    # Save the combined map
    combined_file = os.path.join(output_dir, 'combined_satellite_map.png')
    plt.savefig(combined_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Combined satellite map saved to {combined_file}")
    
    # Return the paths to the generated files
    return {
        'empty_map': empty_file if 'empty_file' in locals() else None,
        'frequency_map': freq_file,
        'confidence_map': conf_file,
        'ground_truth_map': gt_file,
        'combined_map': combined_file
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate maps with satellite background from image retrieval results")
    parser.add_argument("--file_pattern", default="tilt_corrected_*_results.csv", 
                        help="Glob pattern for CSV files (default: tilt_corrected_*_results.csv)")
    parser.add_argument("--threshold", type=float, default=0.3, 
                        help="Distance threshold for matches in km (default: 0.3)")
    parser.add_argument("--output_dir", default="satellite_maps_highres", 
                        help="Output directory for map images (default: satellite_maps_highres)")
    parser.add_argument("--dpi", type=int, default=1200,
                        help="Resolution of output images (default: 1200)")
    parser.add_argument("--zoom", type=int, default=17,
                        help="Zoom level for satellite imagery (default: 17, higher = more detailed)")
    
    args = parser.parse_args()
    
    # Load all CSV files
    combined_df = load_csv_files(args.file_pattern)
    
    # Create all maps
    output_files = create_satellite_maps(
        combined_df, 
        args.output_dir, 
        args.threshold, 
        args.dpi
    )
    
    print(f"\nAll ultra-high-resolution satellite maps have been generated and saved to the {args.output_dir} directory.")
    if output_files.get('empty_map'):
        print(f"- empty_satellite_map.png: Just the satellite imagery with no data points")
    print(f"- frequency_satellite_map.png: Shows how frequently each tile is returned")
    print(f"- confidence_satellite_map.png: Shows average confidence (similarity) for each tile")
    print(f"- ground_truth_satellite_map.png: Shows the best match locations for each UAV image")
    print(f"- combined_satellite_map.png: Combined visualization with both frequency and confidence")
    print(f"\nAll images were generated at {args.dpi} DPI for better zoom capability") 
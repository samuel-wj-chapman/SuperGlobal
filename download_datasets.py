#!/usr/bin/env python3
"""
Dataset download utilities for SuperGlobal.

This script provides functions to download:
1. Oxford and Paris datasets (roxford5k, rparis6k)
2. RevisitOP 1M distractor dataset
3. Pre-computed features

Usage:
    python download_datasets.py --data_dir ./revisitop [--download_distractors] [--download_features]

Original code from the revisitop repository, modified for SuperGlobal.
"""

import os
import urllib.request
import tarfile
import argparse
import pickle
import requests
from tqdm import tqdm
import subprocess
import sys


def download_file(url, destination):
    """Download a file with progress bar"""
    print(f"Downloading {url} -> {destination}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    progress_bar = tqdm(
        total=total_size, 
        unit='iB', 
        unit_scale=True,
        desc=f"Downloading {os.path.basename(destination)}"
    )
    
    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        print(f"ERROR: Download of {url} incomplete")
        return False
    
    return True


def run_command(cmd):
    """Run a shell command and print output"""
    print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False


def download_datasets(data_dir):
    """
    Downloads the necessary datasets for testing.
      
    Arguments:
        data_dir: Path to the data directory
    
    This function checks if the data necessary for running the example script exist.
    If not it downloads it in the folder structure:
        data_dir/datasets/roxford5k/ : folder with Oxford images
        data_dir/datasets/rparis6k/  : folder with Paris images
    """

    # Create data folder if it does not exist
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    
    # Create datasets folder if it does not exist
    datasets_dir = os.path.join(data_dir, 'datasets')
    if not os.path.isdir(datasets_dir):
        os.mkdir(datasets_dir)

    # Download datasets folders datasets/DATASETNAME/
    datasets = ['roxford5k', 'rparis6k']
    for di in range(len(datasets)):
        dataset = datasets[di]

        if dataset == 'roxford5k':
            src_dir = 'https://www.robots.ox.ac.uk/~vgg/data/oxbuildings'
            dl_files = ['oxbuild_images-v1.tgz']
        elif dataset == 'rparis6k':
            src_dir = 'https://www.robots.ox.ac.uk/~vgg/data/parisbuildings'
            dl_files = ['paris_1-v1.tgz', 'paris_2-v1.tgz']
        else:
            raise ValueError('Unknown dataset: {}!'.format(dataset))

        dst_dir = os.path.join(data_dir, 'datasets', dataset, 'jpg')
        if not os.path.isdir(dst_dir):
            print('>> Dataset {} directory does not exist. Creating: {}'.format(dataset, dst_dir))
            os.makedirs(dst_dir)
            for dli in range(len(dl_files)):
                dl_file = dl_files[dli]
                src_file = os.path.join(src_dir, dl_file)
                dst_file = os.path.join(dst_dir, dl_file)
                print('>> Downloading dataset {} archive {}...'.format(dataset, dl_file))
                
                # Use download_file function with progress bar
                download_file(src_file, dst_file)
                
                print('>> Extracting dataset {} archive {}...'.format(dataset, dl_file))
                # create tmp folder
                dst_dir_tmp = os.path.join(dst_dir, 'tmp')
                os.makedirs(dst_dir_tmp, exist_ok=True)
                
                # extract in tmp folder
                run_command('tar -zxf {} -C {}'.format(dst_file, dst_dir_tmp))
                
                # remove all (possible) subfolders by moving only files in dst_dir
                run_command('find {} -type f -exec mv -i {{}} {} \\;'.format(dst_dir_tmp, dst_dir))
                
                # remove tmp folder
                run_command('rm -rf {}'.format(dst_dir_tmp))
                
                print('>> Extracted, deleting dataset {} archive {}...'.format(dataset, dl_file))
                run_command('rm {}'.format(dst_file))

        # Download ground truth file
        gnd_src_dir = os.path.join('http://cmp.felk.cvut.cz/revisitop/data', 'datasets', dataset)
        gnd_dst_dir = os.path.join(data_dir, 'datasets', dataset)
        gnd_dl_file = 'gnd_{}.pkl'.format(dataset)
        gnd_src_file = os.path.join(gnd_src_dir, gnd_dl_file)
        gnd_dst_file = os.path.join(gnd_dst_dir, gnd_dl_file)
        
        if not os.path.exists(gnd_dst_file):
            print('>> Downloading dataset {} ground truth file...'.format(dataset))
            download_file(gnd_src_file, gnd_dst_file)

        # Alternative download source directly from GitHub if the above fails
        if not os.path.exists(gnd_dst_file):
            github_url = f"https://github.com/filipradenovic/revisitop/raw/master/data/datasets/gnd_{dataset}.pkl"
            print(f">> Trying alternative source for {dataset} ground truth file...")
            download_file(github_url, gnd_dst_file)


def download_distractors(data_dir):
    """
    Downloads the distractor dataset.

    Arguments:
        data_dir: Path to the data directory
    
    This function checks if the distractor dataset exists.
    If not it downloads it in the folder:
        data_dir/datasets/revisitop1m/   : folder with 1M distractor images
    """

    # Create data folder if it does not exist
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    
    # Create datasets folder if it does not exist
    datasets_dir = os.path.join(data_dir, 'datasets')
    if not os.path.isdir(datasets_dir):
        os.mkdir(datasets_dir)

    dataset = 'revisitop1m'
    nfiles = 100
    src_dir = 'http://ptak.felk.cvut.cz/revisitop/revisitop1m/jpg'
    dl_files = 'revisitop1m.{}.tar.gz'
    dst_dir = os.path.join(data_dir, 'datasets', dataset, 'jpg')
    dst_dir_tmp = os.path.join(data_dir, 'datasets', dataset, 'jpg_tmp')
    
    if not os.path.isdir(dst_dir):
        print('>> Dataset {} directory does not exist.\n>> Creating: {}'.format(dataset, dst_dir))
        if not os.path.isdir(dst_dir_tmp):
            os.makedirs(dst_dir_tmp)
            
        for dfi in range(nfiles):
            dl_file = dl_files.format(dfi+1)
            src_file = os.path.join(src_dir, dl_file)
            dst_file = os.path.join(dst_dir_tmp, dl_file)
            dst_file_tmp = os.path.join(dst_dir_tmp, dl_file + '.tmp')
            
            if os.path.exists(dst_file):
                print('>> [{}/{}] Skipping dataset {} archive {}, already exists...'.format(dfi+1, nfiles, dataset, dl_file))
            else:
                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    try:
                        print('>> [{}/{}] Downloading dataset {} archive {}...'.format(dfi+1, nfiles, dataset, dl_file))
                        download_file(src_file, dst_file_tmp)
                        os.rename(dst_file_tmp, dst_file)
                        break
                    except Exception as e:
                        retry_count += 1
                        print(f'>>>> Download failed ({retry_count}/{max_retries}): {e}')
                        if retry_count >= max_retries:
                            print(f'>>>> Skipping file after {max_retries} failed attempts')
                
        for dfi in range(nfiles):
            dl_file = dl_files.format(dfi+1)
            dst_file = os.path.join(dst_dir_tmp, dl_file)
            
            if os.path.exists(dst_file):
                print('>> [{}/{}] Extracting dataset {} archive {}...'.format(dfi+1, nfiles, dataset, dl_file))
                try:
                    tar = tarfile.open(dst_file)
                    tar.extractall(path=dst_dir_tmp)
                    tar.close()
                    print('>> [{}/{}] Extracted, deleting dataset {} archive {}...'.format(dfi+1, nfiles, dataset, dl_file))
                    os.remove(dst_file)
                except Exception as e:
                    print(f'>>>> Error extracting {dst_file}: {e}')
        
        # rename tmp folder
        if os.path.exists(dst_dir_tmp) and os.listdir(dst_dir_tmp):
            if os.path.exists(dst_dir):
                # If destination already exists, merge the contents
                for item in os.listdir(dst_dir_tmp):
                    src_item = os.path.join(dst_dir_tmp, item)
                    dst_item = os.path.join(dst_dir, item)
                    if os.path.isfile(src_item):
                        run_command(f'mv {src_item} {dst_item}')
                os.rmdir(dst_dir_tmp)
            else:
                os.rename(dst_dir_tmp, dst_dir)

        # download image list
        gnd_src_dir = 'http://ptak.felk.cvut.cz/revisitop/revisitop1m/'
        gnd_dst_dir = os.path.join(data_dir, 'datasets', dataset)
        gnd_dl_file = '{}.txt'.format(dataset)
        gnd_src_file = os.path.join(gnd_src_dir, gnd_dl_file)
        gnd_dst_file = os.path.join(gnd_dst_dir, gnd_dl_file)
        if not os.path.exists(gnd_dst_file):
            print('>> Downloading dataset {} image list file...'.format(dataset))
            download_file(gnd_src_file, gnd_dst_file)


def download_features(data_dir):
    """
    Downloads pre-computed features for testing.

    Arguments:
        data_dir: Path to the data directory
    
    This function checks if the features necessary for running the example script exist.
    If not it downloads them in the folder: data_dir/features
    """

    # Create data folder if it does not exist
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    
    # Create features folder if it does not exist
    features_dir = os.path.join(data_dir, 'features')
    if not os.path.isdir(features_dir):
        os.mkdir(features_dir)

    # Download example features
    datasets = ['roxford5k', 'rparis6k']
    for di in range(len(datasets)):
        dataset = datasets[di]

        feat_src_dir = os.path.join('http://cmp.felk.cvut.cz/revisitop/data', 'features')
        feat_dst_dir = os.path.join(data_dir, 'features')
        feat_dl_file = '{}_resnet_rsfm120k_gem.mat'.format(dataset)
        feat_src_file = os.path.join(feat_src_dir, feat_dl_file)
        feat_dst_file = os.path.join(feat_dst_dir, feat_dl_file)
        if not os.path.exists(feat_dst_file):
            print('>> Downloading dataset {} features file {}...'.format(dataset, feat_dl_file))
            download_file(feat_src_file, feat_dst_file)


def main():
    parser = argparse.ArgumentParser(description='Download datasets for SuperGlobal')
    parser.add_argument('--data_dir', default='./revisitop', help='Path to data directory')
    parser.add_argument('--download_distractors', action='store_true', help='Download 1M distractor dataset (large download)')
    parser.add_argument('--download_features', action='store_true', help='Download pre-computed features')
    args = parser.parse_args()
    
    print('>> Downloading datasets...')
    download_datasets(args.data_dir)
    
    if args.download_distractors:
        print('>> Downloading distractor dataset (1M)...')
        download_distractors(args.data_dir)
    
    if args.download_features:
        print('>> Downloading pre-computed features...')
        download_features(args.data_dir)
    
    print('>> Done!')


if __name__ == '__main__':
    main() 
import torch
from model.CVNet_Rerank_model import CVNet_Rerank
from test.dataset import DataSet
from tqdm import tqdm
import torch.nn.functional as F
import argparse
import os

@torch.no_grad()
def extract_feature(model, test_loader, scale_list):
    with torch.no_grad():
        img_feats = [[] for i in range(1)] 
        count = 0
        for im_list in tqdm(test_loader):
            if count % 100 == 0:
                print(f"Image Processed {count}")
            count+=1
            
            for idx in range(len(im_list)):
                im_list[idx] = im_list[idx].cuda()
                desc = model.extract_global_descriptor(im_list[idx], True, True, True, scale_list)
                if len(desc.shape) == 1:
                    desc.unsqueeze_(0)
                desc = F.normalize(desc, p=2, dim=1)
                img_feats[idx].append(desc.detach().cpu())
            
        for idx in range(len(img_feats)):
            img_feats[idx] = torch.cat(img_feats[idx], dim=0)
            if len(img_feats[idx].shape) == 1:
                img_feats[idx].unsqueeze_(0)

        img_feats_agg = F.normalize(torch.mean(torch.cat([img_feat.unsqueeze(0) for img_feat in img_feats], dim=0), dim=0), p=2, dim=1)
    
    return img_feats_agg

def main():
    parser = argparse.ArgumentParser(description='Generate Oxford embedding')
    parser.add_argument('--weight', required=True, help='Path to weight')
    parser.add_argument('--depth', default=101, type=int, help='Depth of ResNet')
    parser.add_argument('--data_dir', default='./revisitop', help='Path to revisitop data directory')
    parser.add_argument('--dataset', default='roxford5k', choices=['roxford5k', 'rparis6k'], help='Dataset to use')
    args = parser.parse_args()
    
    weight_path, depth, data_dir, dataset = args.weight, args.depth, args.data_dir, args.dataset
    
    # Create model
    print(f"=> Creating SuperGlobal model with ResNet-{depth}")
    model = CVNet_Rerank(depth, 2048, True)  # True for relup
    
    # Load weights
    print(f"=> Loading weights from {weight_path}")
    weight = torch.load(weight_path)
    
    # Handle different weight formats
    if 'model_state' in weight:
        weight_new = {}
        for i, j in zip(weight['model_state'].keys(), weight['model_state'].values()):
            weight_new[i.replace('globalmodel', 'encoder_q')] = j
        weight = weight_new
    
    model.load_state_dict(weight, strict=False)
    model.cuda()
    model.eval()
    
    # Set up correct ground truth file
    if dataset == 'roxford5k':
        gnd_fn = 'gnd_roxford5k.pkl'
    elif dataset == 'rparis6k':
        gnd_fn = 'gnd_rparis6k.pkl'
    
    # Create dataset and dataloader for database images
    dataset_obj = DataSet(data_dir, dataset, gnd_fn, "db", [1.0])
    dataloader = torch.utils.data.DataLoader(
        dataset_obj,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    
    # Extract features
    print(f"Extracting features from {dataset} dataset...")
    features = extract_feature(model, dataloader, 3)
    
    # Save features
    output_file = f"feats_1m_RN{depth}.pth"
    print(f"Saving features to {output_file}")
    torch.save(features, output_file)
    print(f"Features saved successfully to {output_file}")

if __name__ == "__main__":
    main() 
# written by Shihao Shao (shaoshihao@pku.edu.cn)


import torch
from torch import nn
import numpy as np



class RerankwMDA(nn.Module):
    """ Reranking with maximum descriptors aggregation """
    def __init__(self, M=400, K = 9, beta = 0.15):
        super(RerankwMDA, self).__init__()
        self.M = M 
        self.K = K + 1 # including oneself
        self.beta = beta
    def forward(self, ranks, rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba):
        try:
            # Ensure we're working with detached copies to prevent memory leaks
            ranks_trans_1000 = torch.stack(rerank_dba_final, 0).detach() # 70 400
            ranks_value_trans_1000 = -torch.sort(-res_top1000_dba.detach(), -1)[0] # 70 400
            

            ranks_trans = torch.unsqueeze(ranks_trans_1000_pre[:,:self.K].detach(), -1) # 70 10 1
            ranks_value_trans = torch.unsqueeze(ranks_value_trans_1000[:,:self.K].clone().detach(), -1) # 70 10 1
            ranks_value_trans[:,:,:] *= self.beta
            
            # Make sure the tensors are properly sized before operations
            X1 = torch.take_along_dim(x_dba.detach(), ranks_trans, 1) # 70 10 2048
            X2 = torch.take_along_dim(x_dba.detach(), torch.unsqueeze(ranks_trans_1000_pre.detach(), -1), 1) # 70 400 2048
            
            # Use a more memory-efficient approach for max operation
            X1 = torch.max(X1, 1, keepdim=True)[0] # 70 1 2048
            
            # Split the einsum operation to use less memory
            # Instead of: res_rerank = torch.sum(torch.einsum('abc,adc->abd',X1,X2),1)
            batch_size, num_samples, feat_dim = X2.size()
            res_rerank = torch.zeros(batch_size, num_samples, device=X1.device)
            
            # Process in smaller batches to avoid memory issues
            batch_size = 10  # Adjust this based on available memory
            for i in range(0, X2.size(0), batch_size):
                end_idx = min(i + batch_size, X2.size(0))
                X1_batch = X1[i:end_idx]
                X2_batch = X2[i:end_idx]
                
                # Compute the batch result
                batch_result = torch.bmm(X1_batch.view(end_idx-i, 1, feat_dim), 
                                         X2_batch.transpose(1, 2))  # (batch, 1, num_samples)
                res_rerank[i:end_idx] = batch_result.squeeze(1)

            # Combine results as before
            res_rerank = (ranks_value_trans_1000 + res_rerank) / 2. # 70 400
            res_rerank_ranks = torch.argsort(-res_rerank, axis=-1) # 70 400
            
            rerank_qe_final = []
            ranks_transpose = torch.transpose(ranks.detach(), 1, 0)[:, self.M:] # 70 6322-400
            for i in range(res_rerank_ranks.shape[0]):
                temp_concat = torch.cat([ranks_trans_1000[i][res_rerank_ranks[i]], ranks_transpose[i]], 0)
                rerank_qe_final.append(temp_concat) # 6322
            
            # Clean up intermediate results to free memory
            del X1, X2, res_rerank, ranks_trans_1000, ranks_value_trans_1000
            torch.cuda.empty_cache()
            
            return torch.transpose(torch.stack(rerank_qe_final, 0), 1, 0) # 70 6322
            
        except Exception as e:
            print(f"Error in RerankwMDA forward pass: {e}")
            # Return the original ranks in case of error
            return ranks

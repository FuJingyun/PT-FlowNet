import torch
import torch.nn as nn


from model.corr import new_CorrBlock
from model.update import UpdateBlock
from model.flot.graph import Graph
from model.pointtransformer.pointtransformer_seg import PointTransformerSeg



def B_N_xyz2pxo(input,max_points):
    b, n, xyz = input.size()
    if b>1:
        n_o, count = [ max_points ],max_points
        tmp = [input[0]]
        for i in range(1, b):
            count += max_points
            n_o.append(count)
            tmp.append(input[i])
        n_o = torch.cuda.IntTensor(n_o)
        coord = torch.cat(tmp, 0)  
    else:
        n_o = [ max_points ]
        n_o = torch.cuda.IntTensor(n_o)
        coord = input[0]
    feat = coord
    label =  n_o   
    return coord, feat, label # (n, 3), (n, c), (b)

def pt_fmap_trans(input, max_points):
    p_all, x = input.size()
    b_num = p_all // max_points
    f = input.transpose(0,1)
    f = f.view( x, b_num, p_all // b_num)
    f = f.transpose(0,1)
    return f 

def pt_fmap_trans_ot(input, max_points):
    p_all, x = input.size()
    b_num = p_all // max_points
    f = input.transpose(0,1)
    f = f.view( x, b_num, p_all // b_num)
    f = f.permute(1,2,0)
    return f 
    


class pt_RSF(nn.Module):
    def __init__(self, args):
        super(pt_RSF, self).__init__()
        self.args = args
        self.num_neighbors = 32
        self.hidden_dim = 64 
        self.context_dim = 64
        self.feature_extractor = PointTransformerSeg()
        self.context_extractor = PointTransformerSeg()       
        self.corr_block = new_CorrBlock(num_levels=args.corr_levels, base_scale=args.base_scales,
                                    resolution=3, truncate_k=args.truncate_k,knn = self.num_neighbors)                             
        self.update_block = UpdateBlock(hidden_dim=self.hidden_dim)


    def forward(self, p, num_iters = 12):
        [xyz1, xyz2] = p

        p1, x1, o1 = B_N_xyz2pxo(p[0],self.args.max_points)
        p2, x2, o2 = B_N_xyz2pxo(p[1],self.args.max_points)
        
        fmap1_origin = self.feature_extractor([p1, x1, o1])
        fmap2_origin = self.feature_extractor([p2, x2, o2])
        
        fmap1 = pt_fmap_trans(fmap1_origin,self.args.max_points)
        fmap2 = pt_fmap_trans(fmap2_origin,self.args.max_points)

        # correlation matrix
        self.corr_block.init_module(fmap1, fmap2, xyz2)
        fct1_origin = self.context_extractor([p1, x1, o1])
        fct1 = pt_fmap_trans(fct1_origin,self.args.max_points)
        graph_context = Graph.construct_graph(p[0], self.num_neighbors)

        net, inp = torch.split(fct1, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp) 

        coords1, coords2 = xyz1, xyz1
        flow_predictions = []

        for itr in range(num_iters):
            coords2 = coords2.detach()
            corr = self.corr_block(coords=coords2)
            flow = coords2 - coords1
            net, delta_flow = self.update_block(net, inp, corr, flow, graph_context)
            coords2 = coords2 + delta_flow
            flow_predictions.append(coords2 - coords1)

        return flow_predictions



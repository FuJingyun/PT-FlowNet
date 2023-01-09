import torch.nn as nn

from model.pointtransformer.pointtransformer_seg import new_PTRefine


def pt_flow_trans(input, max_points):
    p_all, x = input.size()
    b_num = p_all // max_points
    f = input.transpose(0,1)
    f = f.view( x, b_num, p_all // b_num)
    f = f.permute(1,2,0)
    return f 


class new_pt_Refine(nn.Module):
    def __init__(self):
        super(new_pt_Refine, self).__init__()
        n = 32
        self.ref_pt = new_PTRefine()
        self.fc = nn.Linear(4 * n, 3)

    def forward(self, pxo, max_points):
        x = self.ref_pt(pxo)
        x = pt_flow_trans(x, max_points)
        x = self.fc(x)
        return  x

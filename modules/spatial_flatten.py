import torch.nn as nn
import torch, math, pdb

class Spatialflatten(nn.Module):
    def __init__(self, amd0=1444, kd=3):
        super(Spatialflatten, self).__init__()
        print('*** Spatialflatten.py : __init__() ***', amd0, kd)
        self.use_gpu = True
        self.amd0 = amd0
        self.kd = kd
        self.padding = nn.ReplicationPad2d(1).cuda()
        with torch.no_grad():
            ww = hh = int(math.sqrt(amd0))
            counts = torch.LongTensor(amd0,kd*kd)
            v = [[(hh+2)*i + j for j in range(ww+2)] for i in range(hh+2)]
            count = 0
            # pdb.set_trace()
            # print(counts.size())
            for h in range(1, hh+1):
                for w in range(1, ww+1):
                    if kd>1:
                        temp =  torch.LongTensor([v[h - 1][w - 1], v[h - 1][w], v[h - 1][w + 1],
                                                            v[h][w - 1], v[h][w], v[h][w + 1],
                                                            v[h + 1][w - 1], v[h + 1][w], v[h + 1][w + 1]])
                    else:
                        temp =  torch.LongTensor([ v[h][w]])

                    # print(temp.min())
                    counts[count, :] = temp
                    count = count + 1

            counts = counts.cuda()
        counts.requires_grad = False
        self.register_buffer("counts", counts)

    def forward(self, fm):
        #print('fm size and max ', fm.size(), torch.max(self.counts))
        fm = self.padding(fm)
        fm = fm.permute(0, 2, 3, 1).contiguous()
        fm = fm.view(fm.size(0), -1, fm.size(3))
        #print('fm size and max ', fm.size(), torch.max(self.counts))
        pfm = fm.index_select(1, self.counts[:,0])
        for h in range(1, self.kd*self.kd):
            temp = fm.index_select(1, self.counts[:, h])
            pfm = torch.cat((pfm, temp),2)
            
        return pfm
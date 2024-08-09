import torch
import torch.nn as nn

anchor = torch.tensor([0.5, -0.5, 0.1])
pos = torch.tensor([0.7, 0.2, 0.1])
neg = torch.tensor([0.8, 0.9, 0.2])

triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
loss = triplet_loss(anchor, pos, neg)
loss.backward()

print(loss)
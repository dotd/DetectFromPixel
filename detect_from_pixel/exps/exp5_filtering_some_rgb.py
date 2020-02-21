import torch

a = torch.tensor([[[10, 11, 12], [9, 10, 11]], [[12, 13, 14], [8, 9, 10]]], dtype=torch.uint8)
print(a)
print("a[0,0]=" + str(a[0, 0]))
print("a[0,1]=" + str(a[0, 1]))
print("a[1,0]=" + str(a[1, 0]))
print("a[1,1]=" + str(a[1, 1]))

idx = (a >= torch.tensor([10, 11, 12], dtype=torch.uint8)) & (a <= torch.tensor([12, 13, 12], dtype=torch.uint8))
idx2 = torch.prod(idx, axis=2)
idx2 = torch.stack((idx2, idx2, idx2), 2)
print("idx double condition=" + str(idx))
print("idx2 double condition=" + str(idx2))
print(a)
print("")
print(torch.where(idx2==True, a, a))
"""
idx = a >= torch.tensor([10], dtype=torch.uint8)
print("idx=" + str(idx))

b = torch.nonzero((a <= torch.tensor([10, 11, 12], dtype=torch.uint8)) & (a >= torch.tensor([8, 9, 10], dtype=torch.int8)))
print(b)
"""
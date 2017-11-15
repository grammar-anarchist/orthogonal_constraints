import torch

def norm(v):
  assert len(v.size())==2
  return v.norm(p=2, dim=1, keepdim=True)

def unit(v, eps=1e-8):
  vnorm = norm(v)
  return v/vnorm.add(eps), vnorm

def xTy(x, y):
  assert len(x.size())==2 and len(y.size())==2,'xTy'
  return torch.sum(x*y, dim=1, keepdim=True)

#def clip_by_norm(v, clip_norm):
#  v_norm = v.norm(p=2, dim=1)
#  for (i, q) in enumerate(v_norm):
#    if q>clip_norm:
#      v[i,...]*=(clip_norm/q)
#  return v
import pdb
def clip_by_norm(v, clip_norm):
  v_norm = norm(v)
  if v.is_cuda:
    scale = torch.ones(v_norm.size()).cuda()
  else:
    scale = torch.ones(v_norm.size())
  mask = v_norm > clip_norm
  scale[mask] = clip_norm/v_norm[mask]

  return v*scale

def gproj(y, g, normalize=False):
  if normalize:
    y,_ = unit(y)

  yTg = xTy(y,g)
  return  g-(yTg*y)

def gexp(y, h, normalize=False):
  if normalize:
    y,_ = unit(y)
    h = gproj(y,h)

  u, hnorm = unit(h)
  return y*hnorm.cos() + u*hnorm.sin()

# parallel translation of tangent vector h1 toward h2
# both h1 and h2 are targent vector on y
def gpt2(y, h1, h2, normalize=False):
  if normalize:
    h1 = gproj(y, h1)
    h2 = gproj(y, h2)

  # h2 = u * sigma  svd of h2
  [u, unorm] = unit(h2)
  uTh1 = xTy(u,h1)
  return h1 - uTh1*( unorm.sin()*y + (1-unorm.cos())*u )


# parallel translation if h1=h2
def gpt(y, h, normalize=False):
  if normalize:
    h = gproj(y, h)

  [u, unorm] = unit(h)
  return (u*unorm.cos() - y*unorm.sin())*unorm

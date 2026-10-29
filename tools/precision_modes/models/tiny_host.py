"""CPU host arithmetic used by the TinyLlama capture recipe."""

import torch


def bf(x):
    return x.to(torch.bfloat16).float()


def fma(a, b, c):
    return (a.double() * b.double() + c.double()).float()


def norm(x, weight, epsilon=1e-5):
    rows, width = x.shape
    assert width == 2048
    squares = x * x
    v = squares.reshape(rows, 64, 4, 8)
    partial = torch.zeros((rows, 4, 8), dtype=torch.float32)
    for block in range(4):
        local = torch.zeros_like(partial)
        for i in range(16):
            local = local + v[:, block * 16 + i]
        partial = partial + local
    lanes = partial[:, 0].clone()
    for group in range(1, 4):
        lanes = lanes + partial[:, group]
    total = torch.zeros((rows,), dtype=torch.float32)
    for lane in range(8):
        total = total + lanes[:, lane]
    inverse = 1.0 / torch.sqrt(total / float(width) + epsilon)
    return bf(bf(x * inverse[:, None]) * weight)


def rotary(x, cosine, sine):
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return bf(bf(x * cosine) + bf(rotated * sine))


def exp_nonpositive(x):
    safe = torch.clamp(x, min=-104.0)
    q = torch.round(safe * torch.tensor(1.4426950408889634,
                                      dtype=torch.float32))
    s = fma(q, torch.tensor(-0.693145751953125), safe)
    s = fma(q, torch.tensor(-1.428606765330187045e-06), s)
    u = torch.full_like(s, 0.000198527617612853646278381)
    for constant in (0.00139304355252534151077271,
                     0.00833336077630519866943359,
                     0.0416664853692054748535156,
                     0.166666671633720397949219, 0.5):
        u = fma(u, s, torch.tensor(constant, dtype=torch.float32))
    u = 1.0 + fma(s * s, u, s)
    qi = q.to(torch.int32)
    half = torch.div(qi, 2, rounding_mode="floor")
    out = torch.ldexp(torch.ldexp(u, half), qi - half)
    return torch.where(x < -104.0, 0.0, out)


def softmax(x):
    shifted = x - x.amax(dim=-1, keepdim=True)
    ex = exp_nonpositive(shifted)
    n = x.shape[-1]
    lanes = torch.zeros((*x.shape[:-1], 16), dtype=torch.float32)
    for i in range(0, n, 16):
        size = min(16, n - i)
        lanes[..., :size] = lanes[..., :size] + ex[..., i:i+size]
    for delta in (8, 4, 2, 1):
        lanes[..., :delta] = lanes[..., :delta] + lanes[..., delta:2*delta]
    reciprocal = 1.0 / lanes[..., :1]
    return bf(ex * reciprocal)


def pair_bmm(a, b, transposed, decode=False):
    # BF16 inputs are represented exactly by FP32 tensors.
    heads, rows, reduction = a.shape
    right = b.transpose(-1, -2) if transposed else b
    columns = right.shape[-1]
    if decode and transposed and reduction == 64:
        lanes = torch.zeros((heads, columns, 16), dtype=torch.float32)
        br = b
        for start in range(0, 64, 16):
            lanes = fma(a[:, 0, start:start+16, None].transpose(-1,-2),
                        br[:, :, start:start+16], lanes)
        for distance in (8,4,2,1):
            lanes = lanes + lanes[..., torch.arange(16) ^ distance]
        return bf(lanes[..., 0].unsqueeze(1))
    total = torch.zeros((heads, rows, columns), dtype=torch.float32)
    if decode and not transposed and columns == 64:
        start = 0
        for block in (8,4,2,1):
            while reduction-start >= block:
                partial = torch.zeros_like(total)
                for pair in range(start,start+block,2):
                    if pair+1 < start+block:
                        partial = fma(a[:,:,pair+1,None],
                                      right[:,None,pair+1,:],partial)
                    partial = fma(a[:,:,pair,None],right[:,None,pair,:],partial)
                total = total + partial
                start += block
    else:
        for pair in range(0,reduction,2):
            if pair+1 < reduction:
                total = fma(a[:,:,pair+1,None],right[:,None,pair+1,:],total)
            total = fma(a[:,:,pair,None],right[:,None,pair,:],total)
    return bf(total)


def attention(q, k, v, cosine, sine, past):
    rows=q.shape[0]
    offset=0 if past is None else past[0].shape[1]
    q=rotary(q.reshape(rows,32,64).transpose(0,1),cosine[offset:offset+rows],
             sine[offset:offset+rows])
    k=rotary(k.reshape(rows,4,64).transpose(0,1),cosine[offset:offset+rows],
             sine[offset:offset+rows])
    v=v.reshape(rows,4,64).transpose(0,1)
    if past is not None:
        k=torch.cat((past[0],k),dim=1)
        v=torch.cat((past[1],v),dim=1)
    rk=k.repeat_interleave(8,dim=0)
    rv=v.repeat_interleave(8,dim=0)
    scores=bf(pair_bmm(q,rk,True,rows==1)*0.125)
    causal=(torch.arange(k.shape[1])[None,:]
            > torch.arange(offset,offset+rows)[:,None])
    scores=bf(scores+torch.where(causal,torch.tensor(-3.3895313892515355e38),
                                 torch.tensor(0.0)))
    probabilities=softmax(scores)
    attended=pair_bmm(probabilities,rv,False,rows==1)
    return attended.transpose(0,1).contiguous().reshape(rows,2048),(k,v)


def silu_mul(gate, up):
    # exp/div execute at FP32, and SiLU rounds before the MLP multiply.
    activated=bf(gate/(1.0+torch.exp(-gate)))
    return bf(activated*up)

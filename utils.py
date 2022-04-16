import torch


def cal_k(hdr_y):
    eps = 1e-8
    I = hdr_y
    Imax = I.max()
    Imin = I.min()
    Iave = torch.exp(torch.log(I + eps).mean())

    A = 0.4
    B = 1.2
    k = A * torch.pow(B, ((Iave + eps).log().mul(2) - (Imin + eps).log() - (Imax + eps).log()) / (
                (Imax + eps).log() - (Imin + eps).log()))
    return k


def cal_tao(hdr_y, k):
    I = hdr_y
    Imax = I.max()
    Imin = I.min()

    tao_0 = 1e-8
    num = 0
    while(True):
        num += 1
        tmp1 = (1 / (I + tao_0) - 1 / (Imin + tao_0)) * ((Imax + tao_0).log() - (Imin + tao_0).log())
        tmp2 = ((I + tao_0).log() - (Imin + tao_0).log()) * (1 / (Imax + tao_0) - 1 / (Imin + tao_0))
        tmp3 = ((Imax + tao_0).log() - (Imin + tao_0).log()).pow(2)
        f_dao = ((tmp1 - tmp2) / tmp3).mean()
        f = (((I + tao_0).log() - (Imin + tao_0).log()) / ((Imax + tao_0).log() - (Imin + tao_0).log())).mean() - k
        tao_1 = tao_0 - (f / f_dao)
        if (tao_0 - tao_1).abs() < 1e-8 or num >= 20:
            if torch.isnan(tao_1):
                exit('tao error!')
            else:
                break
        tao_0 = tao_1
    tao = tao_1
    return tao


def cal_histogram(img_unfold, bins_num):
    local_target_linear = []
    local_target_equlized = []
    for i in range(img_unfold.size(0)):
        target_linear = torch.histc(img_unfold[i, :, :, :].cpu(), bins=bins_num, min=0, max=1)
        target_linear = target_linear.div(target_linear.sum()).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        local_target_linear.append(target_linear)

        target_equlized = torch.ones(bins_num)
        target_equlized = target_equlized.div(target_equlized.sum()).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        local_target_equlized.append(target_equlized)

    return local_target_linear, local_target_equlized


def set_color_map(hdr_y, hdr_r, hdr_g, hdr_b, ldr_y):
    gamma = 0.5
    ldr_r = hdr_r.div(hdr_y).pow(gamma).mul(ldr_y)
    ldr_g = hdr_g.div(hdr_y).pow(gamma).mul(ldr_y)
    ldr_b = hdr_b.div(hdr_y).pow(gamma).mul(ldr_y)
    ldr = torch.cat((ldr_r, ldr_g, ldr_b), 1)
    ldr = torch.clamp(ldr, 0, 1)
    return ldr
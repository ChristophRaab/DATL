import torch


def ltangent_distance(x, y, omegas):
    """
        Compute Orthogonal Complement: math:`\bm P_k = \bm I - \Omega_k \Omega_k^T`
        Compute Tangent Distance: math:`{\| \bm P \bm x - \bm P_k \bm y_k \|}_2`
        :param `torch.tensor` omegas: Three dimensional matrix
        :rtype: `torch.tensor`
        """
    x, y = [arr.view(arr.size(0), -1) for arr in (x, y)]
    p = torch.eye(omegas.shape[-2], device=omegas.device) - torch.bmm(
        omegas, omegas.permute([0, 2, 1]))
    projected_x = x @ p
    projected_y = torch.diagonal(y @ p).T
    expanded_y = torch.unsqueeze(projected_y, dim=1)
    batchwise_difference = expanded_y - projected_x
    differences_squared = batchwise_difference**2
    distances = torch.sqrt(torch.sum(differences_squared, dim=2))
    distances = distances.permute(1, 0)
    return distances


def cosine_similarity(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

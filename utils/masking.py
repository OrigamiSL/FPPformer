import torch


class OffDiagMask_PointLevel():
    def __init__(self, B, V, P, L, device="cpu"):
        with torch.no_grad():
            _mask = torch.eye(L, L, dtype=torch.bool).to(device)
            self._mask = _mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(B, V, P, 1, 1)

    @property
    def mask(self):
        return self._mask


class OffDiagMask_PatchLevel():
    def __init__(self, B, V, P, device="cpu"):
        with torch.no_grad():
            _mask = torch.eye(P, P, dtype=torch.bool).to(device)
            self._mask = _mask.unsqueeze(0).unsqueeze(0).repeat(B, V, 1, 1)

    @property
    def mask(self):
        return self._mask


class TriangularCausalMask():
    def __init__(self, B, V, P, L, device="cpu"):
        with torch.no_grad():
            mask_shape = [L, L]
            _mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
            self._mask = _mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(B, V, P, 1, 1)

    @property
    def mask(self):
        return self._mask

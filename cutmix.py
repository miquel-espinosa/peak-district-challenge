import torch 

def mix_batch_cutmix(batch):
    
    x1, mask1 = batch
    original_x1 = x1.clone() # !!!! copy properly
    permutation = torch.randperm(x1.shape[0])
    x2, mask2 = batch[permutation]
    cutter = BatchCutout(1, (length * x1.size(-1)).round().item(), (length * x1.size(-2)).round().item())
    synthetic_mask = cutter(x1)
    erase_locations = synthetic_mask == 0
    x1[erase_locations] = x1[permutation][erase_locations]
    return x1


class BatchCutout(object):
    """Randomly mask out one or more patches from a batch of images.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        width (int): The width (in pixels) of each square patch.
        height (int): The height (in pixels) of each square patch.
    """
    def __init__(self, n_holes, width, height):
        self.n_holes = n_holes
        self.width = width
        self.height = height

    def __call__(self, img):
        """

        Args:
            img (Tensor): Tensor image of size (B, C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        b = img.size(0)
        c = img.size(1)
        h = img.size(-2)
        w = img.size(-1)

        mask = torch.ones((b, h, w), device=img.device)

        for n in range(self.n_holes):
            y = torch.randint(h, (b,)).long()
            x = torch.randint(w, (b,)).long()

            y1 = (y - self.height // 2).clamp(0, h).int()
            y2 = (y + self.height // 2).clamp(0, h).int()
            x1 = (x - self.width // 2).clamp(0, w).int()
            x2 = (x + self.width // 2).clamp(0, w).int()

            for batch in range(b):
                mask[batch, y1[batch]: y2[batch], x1[batch]: x2[batch]] = 0

        mask = mask.unsqueeze(1).repeat(1, c, 1, 1)

        return mask
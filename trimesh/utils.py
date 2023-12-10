import numpy as np
import trimesh


def bounding_box(mask):
    # Get the x and y coordinates of non-zero values in the mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return np.array([rmin, rmax, cmin, cmax])


def combine_bounding_boxes(bb1, bb2):
    return np.array(
        [
            min(bb1[0], bb2[0]),
            max(bb1[1], bb2[1]),
            min(bb1[2], bb2[2]),
            max(bb1[3], bb2[3]),
        ]
    )


def bbox_to_mask(bbox, shape):
    rmin, rmax, cmin, cmax = bbox
    arr = np.zeros(shape)
    arr[rmin:rmax, cmin:cmax] = 1
    return arr


def get_scale(bb):
    # returns the length of the diagonal of the box
    return np.sqrt((bb[1] - bb[0]) ** 2 + (bb[3] - bb[2]) ** 2)


def get_xy_shift(src_bb, tgt_bb):
    x = (tgt_bb[1] + tgt_bb[0]) - (src_bb[0] + src_bb[1])
    y = (tgt_bb[3] + tgt_bb[2]) - (src_bb[3] + src_bb[2])
    return np.array([x, y]) / 2


def shift_bb_xy(bb, dxy):
    return np.append(bb[:2] + dxy[0], bb[2:] + dxy[1]).astype(int)


def shift_mask(mask, dxy, fval=0.0):
    if not np.any(dxy.astype(int)):
        return np.array(mask)
    shifted = np.ones(mask.shape) * fval
    dx = int(-dxy[1])
    dy = int(-dxy[0])
    if dx < 0 and dy < 0:
        shifted[:dy, :dx] = mask[-dy:, -dx:]
    elif dx < 0 and dy >= 0:
        if dy == 0:
            shifted[:, :dx] = mask[:, -dx:]
        else:
            shifted[dy:, :dx] = mask[:-dy, -dx:]
    elif dx >= 0 and dy < 0:
        if dx == 0:
            shifted[:dy, :] = mask[-dy:, :]
        else:
            shifted[:dy, dx:] = mask[-dy:, :-dx]
    else:
        if dx == 0:
            shifted[dy:, :] = mask[:-dy, :]
        elif dy == 0:
            shifted[:, dx:] = mask[:, :-dx]
        else:
            shifted[dy:, dx:] = mask[:-dy, :-dx]
    return shifted

import os
import sys
import h5py

sys.path.append(os.getcwd())

from freehand.utils import *


def store_as_h5(
    frames,
    fvs,
    tforms,
    tforms_inv,
    acc_transforms_gt,
    inbetween_transforms_gt,
    coords_gt,
    acc_transforms_pred,
    inbetween_transforms_pred,
    coords_pred,
    scan_h5_path
):
    with h5py.File(scan_h5_path, "w") as f:

        grp_frame_data = f.create_group("frame_data")
        grp_gt = f.create_group("gt")
        grp_pred = f.create_group("pred")

        # frame data
        grp_frame_data.create_dataset(
            "frames",
            data=frames,
            compression="gzip",
            chunks=True
        )

        grp_frame_data.create_dataset(
            "fvs",
            data=fvs,
            compression="gzip",
            chunks=True
        )

        # GT data
        grp_gt.create_dataset(
            "tforms",
            data=tforms,
            compression="gzip",
            chunks=True
        )

        grp_gt.create_dataset(
            "tforms_inv",
            data=tforms_inv,
            compression="gzip",
            chunks=True
        )

        grp_gt.create_dataset(
            "acc_transforms_gt",
            data=acc_transforms_gt,
            compression="gzip",
            chunks=True
        )

        grp_gt.create_dataset(
            "inbetween_transforms_gt",
            data=inbetween_transforms_gt,
            compression="gzip",
            chunks=True
        )

        grp_gt.create_dataset(
            "coords_gt",
            data=coords_gt,
            compression="gzip",
            chunks=True
        )

        # prediction data
        grp_pred.create_dataset(
            "acc_transforms_pred",
            data=acc_transforms_pred,
            compression="gzip",
            chunks=True
        )

        grp_pred.create_dataset(
            "inbetween_transforms_pred",
            data=inbetween_transforms_pred,
            compression="gzip",
            chunks=True
        )

        grp_pred.create_dataset(
            "coords_pred",
            data=coords_pred,
            compression="gzip",
            chunks=True
        )

    print(f"Saved scan h5: {scan_h5_path}")
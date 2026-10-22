"""One-sample CT-RATE GPU smoke test for the AnyMC3D adaptation."""

import os

import torch

from data_modules.ct_rate_dataset import CTRateDataset, DEFAULT_LABEL_COLS
from model_arch.anymc3d import AnyMC3DLightningModule


DATASET_ROOT = "<CT_RATE_ROOT>"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the allocated GPU job")

    dataset = CTRateDataset(
        root=f"{DATASET_ROOT}/train_fixed",
        labels_path=f"{DATASET_ROOT}/multi_abnormality_labels/train_predicted_labels.csv",
        exclusion_path=f"{DATASET_ROOT}/metadata/no_chest_train.txt",
        label_cols=DEFAULT_LABEL_COLS,
        num_slices=8,
        image_size=224,
        limit=1,
    )
    volume, labels, volume_name = dataset[0]

    model = AnyMC3DLightningModule(
        task="multilabel",
        num_classes=18,
        modalities=["ct"],
        backbone_name="dinov2_vitb14",
        lora_rank=8,
        lora_alpha=16,
        input_size=224,
        cls_head_dropout=0.1,
        slice_axis=3,
        vision_blocks=0,
        use_slice_attn_pool=True,
        use_patch_concat=False,
        use_25d=False,
        use_patch_attn_pool=False,
        max_epochs=1,
    ).cuda()
    model.train()

    batch = (volume.unsqueeze(0).cuda(non_blocking=True), labels.unsqueeze(0).cuda(), [volume_name])
    loss, probabilities, targets = model._shared_step(batch)
    loss.backward()

    print(f"volume={volume_name}")
    print(f"volume_shape={tuple(volume.shape)}")
    print(f"targets_shape={tuple(targets.shape)}")
    print(f"probabilities_shape={tuple(probabilities.shape)}")
    print(f"loss={loss.item():.6f}")
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"max_gpu_memory_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}")
    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    os.environ.setdefault("WANDB_MODE", "offline")
    main()

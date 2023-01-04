import os
from pathlib import Path

import torch


def load_checkpoint_for_inference(filename, model, ignore_missed=False):
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        if checkpoint:
            miss_keys, _ = model.load_state_dict(
                checkpoint["model_state_dict"], strict=not ignore_missed
            )
            if miss_keys and not ignore_missed:
                print("Can not load network some keys are missing:")
                print(miss_keys)
                exit(-1)
            print(f"Checkpoint {filename} was successfully loaded")
            return True
    print(f"Failed to load checkpoint: {filename} ")
    return False


def load_last_checkpoint(path, model, optimizer, scaler):
    if os.path.exists(path):
        files = list(Path(path).glob("*.pt"))
        if len(files) > 0:

            def file_number(f):
                f = str(f.stem)
                key = f.rsplit("_", 1)[1]
                if key.isdigit():
                    return int(key)
                else:
                    return -1

            files = sorted(files, reverse=True, key=file_number)
            filename = files[0]
            return load_checkpoint(filename, model, optimizer, scaler)
    return -1


def load_checkpoint(filename, model, optimizer, scaler):
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        if checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scaler is not None and "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            epoch = checkpoint["epoch"]
            print(f"Checkpoint {filename} was successfully loaded")
            return epoch
    return -1


def save_checkpoint(name, epoch, model, optimizer, scaler, path):
    Path(path).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(path, f"{name}_{epoch}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        },
        filename,
    )
    print(f"Checkpoint {filename} was successfully saved")

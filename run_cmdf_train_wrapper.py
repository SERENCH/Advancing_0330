import os
import runpy
import sys

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.dataloader


def patch_dataloader_num_workers():
    original_dataloader = torch.utils.data.dataloader.DataLoader

    class PatchedDataLoader(original_dataloader):
        def __init__(self, *args, **kwargs):
            requested_num_workers = kwargs.get("num_workers", 0)
            kwargs["num_workers"] = 0
            if requested_num_workers != 0:
                print(
                    f"[wrapper] Forcing DataLoader num_workers=0 "
                    f"(requested {requested_num_workers}) for Windows stability.",
                    flush=True,
                )
            super().__init__(*args, **kwargs)

    torch.utils.data.DataLoader = PatchedDataLoader
    torch.utils.data.dataloader.DataLoader = PatchedDataLoader


def patch_load_state_dict():
    original_load_state_dict = nn.Module.load_state_dict

    def patched_load_state_dict(self, state_dict, strict=True):
        try:
            return original_load_state_dict(self, state_dict, strict=strict)
        except RuntimeError as exc:
            error_text = str(exc)
            eca_keys = [
                key
                for key in state_dict.keys()
                if "eca_fusion" in key
            ]
            if not eca_keys:
                raise

            print(
                "[wrapper] Retrying load_state_dict with strict=False after "
                f"dropping transient ECA keys: {eca_keys}",
                flush=True,
            )
            filtered_state_dict = {
                key: value
                for key, value in state_dict.items()
                if "eca_fusion" not in key
            }
            return original_load_state_dict(self, filtered_state_dict, strict=False)

    nn.Module.load_state_dict = patched_load_state_dict


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python run_cmdf_train_wrapper.py <train.py path> [train.py args...]"
        )

    train_script = os.path.abspath(sys.argv[1])
    train_args = sys.argv[2:]
    repo_root = os.path.dirname(train_script)

    patch_dataloader_num_workers()
    patch_load_state_dict()

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    sys.argv = [train_script, *train_args]
    runpy.run_path(train_script, run_name="__main__")


if __name__ == "__main__":
    main()

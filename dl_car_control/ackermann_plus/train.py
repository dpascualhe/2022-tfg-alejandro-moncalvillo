import argparse
from datetime import datetime
import json
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,
    vgg11,
    VGG11_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
    vit_b_16,
    ViT_B_16_Weights,
    swin_v2_s,
    Swin_V2_S_Weights,
)
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

from dl_car_control.Ackermann.utils.pilot_net_dataset import PilotNetDataset


def parse_args() -> argparse.Namespace:
    """Parse user input arguments

    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="dl_car_control/ackermann_plus/config.json",
        help="Path to JSON config file",
    )
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    cfg = json.load(open(args.cfg, "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(cfg["outdir"], f"{cfg['run_name']}-{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    # Init model and adapt last layer
    if cfg["backbone"] == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, cfg["num_outputs"])
    elif cfg["backbone"] == "efficientnet_v2_s":
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        model.classifier[-1] = torch.nn.Linear(
            model.classifier[-1].in_features, cfg["num_outputs"]
        )
    elif cfg["backbone"] == "vgg11":
        model = vgg11(weights=VGG11_Weights.DEFAULT)
        model.classifier[-1] = torch.nn.Linear(
            model.classifier[-1].in_features, cfg["num_outputs"]
        )
    elif cfg["backbone"] == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[-1] = torch.nn.Linear(
            model.classifier[-1].in_features, cfg["num_outputs"]
        )
    elif cfg["backbone"] == "vit_b_16":
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model.heads.head = torch.nn.Linear(
            model.heads.head.in_features, cfg["num_outputs"]
        )
    elif cfg["backbone"] == "swin_v2_s":
        model = swin_v2_s(weights=Swin_V2_S_Weights.DEFAULT)
        model.head = torch.nn.Linear(model.head.in_features, cfg["num_outputs"])
    else:
        raise ValueError(f"Invalid backbone {cfg['backbone']}")
    model.to(device)

    # Init transforms
    all_transforms = [transforms.ToImage()]
    if "ColorJitter" in cfg["augmentations"]:
        all_transforms.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=cfg["augmentations"]["ColorJitter"]["brightness"],
                        contrast=cfg["augmentations"]["ColorJitter"]["contrast"],
                        saturation=cfg["augmentations"]["ColorJitter"]["saturation"],
                        hue=cfg["augmentations"]["ColorJitter"]["hue"],
                    )
                ],
                p=cfg["augmentations"]["ColorJitter"]["p"],
            )
        )
    if "GaussianBlur" in cfg["augmentations"]:
        all_transforms.append(
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=cfg["augmentations"]["GaussianBlur"]["kernel_size"],
                        sigma=cfg["augmentations"]["GaussianBlur"]["sigma"],
                    )
                ],
                p=cfg["augmentations"]["GaussianBlur"]["p"],
            )
        )
    if "GaussianNoise" in cfg["augmentations"]:
        all_transforms.append(
            transforms.RandomApply(
                [
                    transforms.GaussianNoise(
                        mean=cfg["augmentations"]["GaussianNoise"]["mean"],
                        sigma=cfg["augmentations"]["GaussianNoise"]["sigma"],
                    )
                ],
                p=cfg["augmentations"]["GaussianNoise"]["p"],
            )
        )
    if "RandomErasing" in cfg["augmentations"]:
        transforms.RandomErasing(
            p=cfg["augmentations"]["RandomErasing"]["p"],
            scale=cfg["augmentations"]["RandomErasing"]["scale"],
            ratio=cfg["augmentations"]["RandomErasing"]["ratio"],
        )

    all_transforms.append(transforms.ToDtype(torch.float32, scale=True))
    all_transforms = transforms.Compose(all_transforms)

    # Init datasets and dataloaders
    train_dataset = PilotNetDataset(
        cfg["train_datasets"],
        flip_images="flip" in cfg["preprocessing"],
        transforms=all_transforms,
        preprocessing=cfg["preprocessing"],
    )

    if "val_datasets" in cfg:
        val_dataset = PilotNetDataset(
            cfg["val_datasets"],
            flip_images=False,
            transforms=transforms.Compose(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                ]
            ),
            preprocessing=None,
        )
    elif "val_split" in cfg:
        num_samples = len(train_dataset)
        num_samples_val = round(num_samples * cfg["val_split"])
        num_samples_train = num_samples - num_samples_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [num_samples_train, num_samples_val]
        )
    else:
        raise Exception("No validation dataset or split provided")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=True,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Init optimizer and loss function
    if cfg["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    else:
        raise ValueError(f"Invalid optimizer {cfg['optimizer']}")

    if cfg["loss"] == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        raise ValueError(f"Invalid loss {cfg['loss']}")

    # Init tensorboard summary
    writer = SummaryWriter(outdir)
    writer.add_text("Configuration", json.dumps(cfg, indent=4))

    # Train model
    best_val_loss = 10**10
    vals_no_improvement = 0
    for epoch_idx in range(cfg["max_epochs"]):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_dataloader, leave=True)
        for batch_idx, (inputs, labels) in enumerate(pbar):
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = loss_fn(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.cpu().item()
            current_loss = running_loss / (batch_idx + 1)

            pbar.set_description(f"TRAIN-Epoch [{epoch_idx + 1}/{cfg['max_epochs']}]")
            pbar.set_postfix(loss=current_loss)

        writer.add_scalar(f"Training loss ({cfg['loss']})", current_loss, epoch_idx)

        save_model = epoch_idx % cfg["save_frequency"] == 0

        if epoch_idx % cfg["val_frequency"] == 0:
            model.eval()
            running_val_loss = 0.0
            pbar = tqdm(val_dataloader, leave=True)
            for batch_idx, (inputs, labels) in enumerate(pbar):
                with torch.no_grad():
                    outputs = model(inputs.to(device))
                    loss = loss_fn(outputs, labels.to(device))

                    running_val_loss += loss.cpu().item()
                    current_val_loss = running_val_loss / (batch_idx + 1)

                    pbar.set_description(
                        f"VAL-Epoch [{epoch_idx + 1}/{cfg['max_epochs']}]"
                    )
                    pbar.set_postfix(loss=current_val_loss)

            writer.add_scalar(
                f"Validation loss ({cfg['loss']})", current_val_loss, epoch_idx
            )
            print(f"Validation loss: {current_val_loss}\n")

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                print(f"New best validation loss: {best_val_loss}")
                save_model = True
                vals_no_improvement = 0
            else:
                vals_no_improvement += 1

            if vals_no_improvement >= cfg["early_stopping_patience"]:
                print(f"Stopping early after {vals_no_improvement} validations without improvement")
                break

        if save_model:
            fname = os.path.join(
                outdir, f"model-epoch_{epoch_idx:03d}-loss_{current_val_loss:.3f}.pth"
            )
            print(f"Saving model to {fname}")
            torch.save(model.state_dict(), fname)

        writer.flush()

    writer.close()


if __name__ == "__main__":
    main()

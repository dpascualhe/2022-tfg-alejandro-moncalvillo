import argparse
import csv
import os
from time import time

import cv2
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2 as transforms
from torchvision.models import (
    resnet18,
    efficientnet_v2_s,
    vgg11,
    mobilenet_v3_small,
    vit_b_16,
    swin_v2_s,
)
from tqdm import tqdm

from dl_car_control.Ackermann.utils.pilotnet import PilotNet


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--cropped", action="store_true")

    args = parser.parse_args()
    return args


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_w_gt = []
    all_w_pred = []
    all_v_gt = []
    all_v_pred = []
    all_t = []

    args = parse_args()
    num_labels = 2
    image_shape = (66, 200, 3)

    if args.model_type == "pilotnet":
        model = PilotNet(image_shape, num_labels)
    elif args.model_type == "resnet18":
        model = resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, num_labels)
    elif args.model_type == "efficientnet_v2_s":
        model = efficientnet_v2_s()
        model.classifier[-1] = torch.nn.Linear(
            model.classifier[-1].in_features, num_labels
        )
    elif args.model_type == "vgg11":
        model = vgg11()
        model.classifier[-1] = torch.nn.Linear(
            model.classifier[-1].in_features, num_labels
        )
    elif args.model_type == "mobilenet_v3_small":
        model = mobilenet_v3_small()
        model.classifier[-1] = torch.nn.Linear(
            model.classifier[-1].in_features, num_labels
        )
    elif args.model_type == "vit_b_16":
        model = vit_b_16()
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_labels)
    elif args.model_type == "swin_v2_s":
        model = swin_v2_s()
        model.head = torch.nn.Linear(model.head.in_features, num_labels)
    else:
        raise ValueError(f"Invalid model type {args.model_type}")

    model.load_state_dict(
        torch.load(args.model, weights_only=True, map_location=device)
    )
    model.to(device)
    model.eval()

    preprocess = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    f = open(os.path.join(args.test, "data.csv"), "r")
    reader_csv = csv.reader(f)
    next(reader_csv, None)  # skip the headers
    samples = [sample for sample in reader_csv]
    num_samples = len(samples)

    total_time = 0
    min_dt = 20000
    max_dt = -1

    total_loss_v = 0
    total_loss_w = 0
    pbar = tqdm(enumerate(samples), total=num_samples, leave=True)
    for idx, (image_name, v_gt, w_gt) in pbar:
        start_time = time()

        image = cv2.imread(os.path.join(args.test, image_name))[:, :, ::-1]
        if args.cropped:
            image = image[240:480, 0:640]
        resized_image = cv2.resize(image, (image_shape[1], image_shape[0]))

        input_tensor = preprocess(resized_image).to(device)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)
            v_pred, w_pred = output.squeeze().cpu().numpy()

        all_v_gt.append(float(v_gt))
        all_w_gt.append(float(w_gt))

        all_v_pred.append(v_pred)
        all_w_pred.append(w_pred)

        total_loss_v = total_loss_v + abs(float(v_gt) - v_pred)
        total_loss_w = total_loss_w + abs(float(w_gt) - w_pred)

        all_t.append(idx)

        finish_time = time()
        dt = finish_time - start_time
        total_time = total_time + dt
        if dt < min_dt:
            min_dt = dt
        if dt > max_dt:
            max_dt = dt

    f.close()

    print("Avg. dt:" + str(total_time / num_samples))
    print("Min. dt:" + str(min_dt))
    print("Max. dt:" + str(max_dt))
    print("Avg. W abs(diff):" + str(total_loss_w / num_samples))
    print("Avg. V abs(diff):" + str(total_loss_v / num_samples))

    plt.subplot(1, 2, 1)
    plt.plot(all_t, all_v_gt, label="controller", color="b")
    plt.plot(all_t, all_v_pred, label="net", color="tab:orange")
    plt.title("Linear speed comparison")
    plt.xlabel("Samples")
    plt.ylabel("Linear speed output")
    plt.legend(loc="upper left")
    plt.subplot(1, 2, 2)
    plt.plot(all_t, all_w_gt, label="controller", color="b")
    plt.plot(all_t, all_w_pred, label="net", color="tab:orange")
    plt.title("Angular speed comparison")
    plt.xlabel("Samples")
    plt.ylabel("Angular speed output")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()

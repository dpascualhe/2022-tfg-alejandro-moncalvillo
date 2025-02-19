import argparse

import cv2
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

import dl_car_control.Ackermann.utils.hal as HAL
from dl_car_control.Ackermann.utils.pilotnet import PilotNet


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--cropped", action="store_true")

    args = parser.parse_args()
    return args


args = parse_args()
num_labels = 2
image_shape = (66, 200, 3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model_type == "pilotnet":
    model = PilotNet(image_shape, num_labels)
elif args.model_type == "resnet18":
    model = resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_labels)
elif args.model_type == "efficientnet_v2_s":
    model = efficientnet_v2_s()
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_labels)
elif args.model_type == "vgg11":
    model = vgg11()
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_labels)
elif args.model_type == "mobilenet_v3_small":
    model = mobilenet_v3_small()
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_labels)
elif args.model_type == "vit_b_16":
    model = vit_b_16()
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_labels)
elif args.model_type == "swin_v2_s":
    model = swin_v2_s()
    model.head = torch.nn.Linear(model.head.in_features, num_labels)
else:
    raise ValueError(f"Invalid model type {args.model_type}")

model.load_state_dict(torch.load(args.model, weights_only=True, map_location=device))
model.to(device)
model.eval()


def user_main():
    image = HAL.getImage()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height = image.shape[0]

    preprocess = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    # crop image
    if height > 100:
        if args.cropped:
            image = image[240:480, 0:640]

        resized_image = cv2.resize(image, (image_shape[1], image_shape[0]))

        input_tensor = preprocess(resized_image).to(device)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)
            v, w = output.squeeze().cpu().numpy()

        HAL.setV(v)
        HAL.setW(w)


def main():

    HAL.setW(0)
    HAL.setV(0)
    HAL.main(user_main)


if __name__ == "__main__":
    main()

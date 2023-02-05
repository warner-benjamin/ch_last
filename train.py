## Author: Thomas Capelle, Soumik Rakshit
## Mail:   tcapelle@wandb.com, soumik.rakshit@wandb.com

import math, argparse
from types import SimpleNamespace
from time import perf_counter

import numpy as np

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
from torch.cuda.amp import autocast

PROJECT = "subclass_tensors"
ENTITY = None
WANDB = False

try:
    import wandb
    WANDB = True
except:
    pass

config_defaults = SimpleNamespace(
    batch_size=96,
    steps=300,
    device="cuda",
    epochs=1,
    num_experiments=1,
    learning_rate=1e-3,
    image_size=224,
    model_name="resnet50",
    num_workers=4,
    mixed_precision=True,
    channels_last=False,
    optimizer="Adam",
    opt_foreach=True,
    subclass=False,
    pt_compile=False,
    log_wandb=False,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, default=ENTITY)
    parser.add_argument('--batch_size', type=int, default=config_defaults.batch_size)
    parser.add_argument('--steps', type=int, default=config_defaults.steps)
    parser.add_argument('--epochs', type=int, default=config_defaults.epochs)
    parser.add_argument('--num_experiments', type=int, default=config_defaults.num_experiments)
    parser.add_argument('--learning_rate', type=float, default=config_defaults.learning_rate)
    parser.add_argument('--image_size', type=int, default=config_defaults.image_size)
    parser.add_argument('--model_name', type=str, default=config_defaults.model_name)
    parser.add_argument('--device', type=str, default=config_defaults.device)
    parser.add_argument('--num_workers', type=int, default=config_defaults.num_workers)
    parser.add_argument('--mixed_precision', type=bool, default=config_defaults.mixed_precision)
    parser.add_argument('--channels_last', type=bool, default=config_defaults.channels_last)
    parser.add_argument('--optimizer', type=str, default=config_defaults.optimizer)
    parser.add_argument('--opt_foreach', type=bool, default=config_defaults.opt_foreach)
    parser.add_argument('--subclass', type=bool, default=config_defaults.subclass)
    parser.add_argument('--pt_compile', type=bool, default=config_defaults.pt_compile)
    parser.add_argument('--log_wandb', type=bool, default=config_defaults.log_wandb)
    return parser.parse_args()


class SubClassedTensor(torch.Tensor):
    pass


class FakeData(tv.datasets.vision.VisionDataset):
    """A fake dataset that returns randomly generated images and returns them as PIL images

    Args:
        size (int, optional): Size of the dataset. Default: 1000 images
        image_size(tuple, optional): Size if the returned images. Default: (3, 224, 224)
        num_classes(int, optional): Number of classes in the dataset. Default: 10
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        random_offset (int): Offsets the index-based random seed used to
            generate each image. Default: 0

    """

    def __init__(
        self,
        size = 1000,
        image_size = (3, 224, 224),
        num_classes = 10,
        random_offset = 0,
    ) -> None:
        super().__init__(None, transform=None, target_transform=None)  # type: ignore[arg-type]
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        self.random_offset = random_offset

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # create random image that is consistent with the index id
        if index >= len(self):
            raise IndexError(f"{self.__class__.__name__} index out of range")
        rng_state = torch.get_rng_state()
        torch.manual_seed(index + self.random_offset)
        img = torch.randn(*self.image_size)
        target = torch.randint(0, self.num_classes, size=(1,), dtype=torch.long)[0]
        torch.set_rng_state(rng_state)
        return img, target.item()

    def __len__(self) -> int:
        return self.size


def get_dataloader(batch_size, image_size=224, num_workers=0, steps=300, **kwargs):
    "Get a training dataloader"
    ds = FakeData(size=batch_size*steps, image_size=(3, image_size, image_size))
    loader = torch.utils.data.DataLoader(ds,
                                         batch_size=batch_size,
                                         pin_memory=True,
                                         num_workers=num_workers,
                                         **kwargs)
    return loader


def get_model(n_out=10, arch="resnet18", weights=None):
    model = getattr(tv.models, arch)(weights=weights, num_classes=n_out)
    return model


def check_cuda(config):
    if torch.cuda.is_available():
        config.device = "cuda"
        config.gpu_name = torch.cuda.get_device_name()
    return config


def train(config=config_defaults):
    log = WANDB and config.log_wandb

    config = check_cuda(config)
    if log:
        wandb.init(project=PROJECT, entity=args.entity, group=args.group, config=config)
    else:
        metrics = []

    # Get the data
    train_dl = get_dataloader(batch_size=config.batch_size,
                              image_size=config.image_size,
                              num_workers=config.num_workers,
                              steps=config.steps)
    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

    model = get_model(arch=config.model_name)
    if config.channels_last:
        model.to(memory_format=torch.channels_last)
    model.to(config.device)
    if config.pt_compile:
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)

    if config.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()

    # Make the loss and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, config.optimizer)
    optimizer = optimizer(model.parameters(), lr=config.learning_rate, foreach=config.opt_foreach)

    # Training
    example_ct = 0
    step_ct = 0
    total_time = perf_counter()
    for epoch in range(config.epochs):
        t0 = perf_counter()
        model.train()

        for step, (images, labels) in enumerate(train_dl):
            if config.subclass:
                images = images.as_subclass(SubClassedTensor)

            images, labels = images.to(config.device), labels.to(config.device)

            if config.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            ti = perf_counter()
            if config.mixed_precision:
                with autocast():
                    outputs = model(images)
                    train_loss = loss_func(outputs, labels)
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                train_loss = loss_func(outputs, labels)
                train_loss.backward()
                optimizer.step()

            optimizer.zero_grad()

            tf = perf_counter()
            example_ct += len(images)
            metric = {"train/train_loss": train_loss.item(),
                      "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
                      "train/example_ct": example_ct,
                      "train_samples_per_sec": len(images)/(tf-ti)}

            if log:
                if step + 1 < n_steps_per_epoch:
                    # 🐝 Log train metrics to wandb
                    wandb.log(metric)
            metrics.append(metric)
            step_ct += 1

    if log:
        np_metrics = np.array([list(m.values()) for m in metrics])
        wandb.summary["total_time"] = perf_counter() - total_time
        wandb.summary["samples_sec_mean"] = np_metrics[:, 3].mean()
        wandb.summary["samples_sec_stddev"] = np_metrics[:, 3].std()
    else:
        print(f'\nRun: PyTorch={torch.__version__}, subclass={config.subclass}, channels_last={config.channels_last}, compile={config.pt_compile}\n')
        print(f'Total Time: {perf_counter() - total_time:.2f}')
        np_metrics = np.array([list(m.values()) for m in metrics])
        print(f'Samples/Second:    Mean: {np_metrics[:, 3].mean():.2f}')
        print(f'Samples/Second: Std Dev: {np_metrics[:, 3].std():.2f}\n')


if __name__ == "__main__":
    args = parse_args()
    for _ in range(args.num_experiments):
        train(config=args)

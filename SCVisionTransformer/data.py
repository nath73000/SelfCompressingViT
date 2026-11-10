from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from SCVisionTransformer.config import DataConfig


DATASETS = {
    "mnist":        datasets.MNIST,
    "fashionmnist": datasets.FashionMNIST,
    "cifar10":      datasets.CIFAR10,
}


def build_dataloaders(cfg: DataConfig, root: str = "./data"):
    if cfg.name not in DATASETS:
        raise ValueError(f"data.name={cfg.name!r} unknown, available: {', '.join(DATASETS)}")
    dataset_cls = DATASETS[cfg.name]

    train_set = dataset_cls(root, train=True,  download=True, transform=transforms.ToTensor())
    test_set  = dataset_cls(root, train=False, download=True, transform=transforms.ToTensor())
    _check_matches_config(train_set, cfg)

    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)])
    train_set.transform = test_set.transform = normalize

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=cfg.eval_batch_size)
    return train_loader, test_loader


def _check_matches_config(dataset, cfg: DataConfig):
    """image_shape and num_classes are written by hand in the YAML: check them against the real data"""
    x, _ = dataset[0]
    if tuple(x.shape) != cfg.image_shape:
        raise ValueError(f"data.image_shape={cfg.image_shape}, but {cfg.name} gives {tuple(x.shape)}")
    if len(dataset.classes) != cfg.num_classes:
        raise ValueError(f"data.num_classes={cfg.num_classes}, but {cfg.name} has {len(dataset.classes)} classes")
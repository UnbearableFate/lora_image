import os
from typing import Dict, List, Optional, Tuple, Union

from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, IterableDataset, load_dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoImageProcessor
from PIL import Image as PILImage

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

VTAB_ALIASES: Dict[str, str] = {
    "caltech101": "fw407/vtab-1k_caltech101",
    "cifar": "fw407/vtab-1k_cifar",
    "dtd": "fw407/vtab-1k_dtd",
    "flowers102": "fw407/vtab-1k_flowers102",
    "pets": "fw407/vtab-1k_pets",
    "svhn": "fw407/vtab-1k_svhn",
    "sun397": "fw407/vtab-1k_sun397",
    "patch_camelyon": "fw407/vtab-1k_patch_camelyon",
    "eurosat": "fw407/vtab-1k_eurosat",
    "resisc45": "fw407/vtab-1k_resisc45",
    "retinopathy": "fw407/vtab-1k_retinopathy",
    "clevr_count": "fw407/vtab-1k_clevr_count",
    "clevr_distance": "fw407/vtab-1k_clevr_distance",
    "dsprites_location": "fw407/vtab-1k_dsprites_location",
    "dsprites_orientation": "fw407/vtab-1k_dsprites_orientation",
    "smallnorb_azimuth": "fw407/vtab-1k_smallnorb_azimuth",
    "smallnorb_elevation": "fw407/vtab-1k_smallnorb_elevation",
    "kitti_distance": "fw407/vtab-1k_kitti_distance",
}


def resolve_dataset_id(name: str) -> str:
    """Return a full HF dataset id given a short alias or passthrough name."""
    return VTAB_ALIASES.get(name, name)

MEDMNIST_PREFIXES = ("medmnist:", "medmnist-v2:")
DEFAULT_MEDMNIST_SIZE = 224


def is_medmnist_dataset_name(dataset_name: str) -> bool:
    return dataset_name.startswith(MEDMNIST_PREFIXES)


def _parse_medmnist_spec(dataset_name: str) -> Tuple[str, int]:
    spec = dataset_name
    for prefix in MEDMNIST_PREFIXES:
        if dataset_name.startswith(prefix):
            spec = dataset_name[len(prefix):]
            break
    if not spec:
        raise ValueError("MedMNIST dataset name is missing a task, e.g. 'medmnist:pathmnist'.")
    if "@" in spec:
        task, size_str = spec.split("@", 1)
        if not size_str:
            raise ValueError(f"MedMNIST spec '{dataset_name}' has an empty size suffix.")
        size = int(size_str)
    else:
        task = spec
        size = DEFAULT_MEDMNIST_SIZE
    task = task.strip().lower()
    if not task:
        raise ValueError("MedMNIST task name cannot be empty.")
    return task, size


def _medmnist_label_names(info_label: Dict[str, str]) -> List[str]:
    try:
        sorted_items = sorted(info_label.items(), key=lambda item: int(item[0]))
    except (TypeError, ValueError):
        sorted_items = list(info_label.items())
    return [name for _, name in sorted_items]


def _to_pil_image(img):
    if hasattr(img, "convert"):
        return img
    if isinstance(img, PILImage.Image):
        return img
    return PILImage.fromarray(img)


def load_medmnist_dataset(
    dataset_name: str,
    cache_dir: Optional[str] = None,
) -> DatasetDict:
    try:
        import medmnist
    except ImportError as exc:
        raise ImportError(
            "MedMNIST support requires the `medmnist` package. "
            "Install it with `pip install medmnist`."
        ) from exc

    task, size = _parse_medmnist_spec(dataset_name)
    info = medmnist.INFO.get(task)
    if info is None:
        available = ", ".join(sorted(medmnist.INFO.keys()))
        raise ValueError(f"Unknown MedMNIST task '{task}'. Available tasks: {available}")

    python_class = info.get("python_class")
    if not python_class or not hasattr(medmnist, python_class):
        raise ValueError(f"MedMNIST task '{task}' does not expose a python_class entry.")

    if info.get("task") == "multi-label, binary-class":
        raise ValueError(
            f"MedMNIST task '{task}' is multi-label; this training pipeline expects single-label classification."
        )

    cls = getattr(medmnist, python_class)
    root = os.path.join(cache_dir or "data_cache", "medmnist")
    label_names = _medmnist_label_names(info.get("label", {}))
    features = Features(
        {
            "image": Image(),
            "label": ClassLabel(names=label_names) if label_names else ClassLabel(num_classes=int(info["n_classes"])),
        }
    )

    splits = {}
    for split in ("train", "val", "test"):
        dataset = cls(split=split, download=True, size=size, root="data")
        images = dataset.imgs
        labels = dataset.labels

        if hasattr(images, "ndim") and images.ndim >= 4 and images.shape[-1] not in (1, 3):
            raise ValueError(
                f"MedMNIST task '{task}' provides 3D volumes; this pipeline expects 2D images."
            )

        labels = labels.astype("int64").squeeze()
        if labels.ndim != 1:
            raise ValueError(f"Unexpected label shape for MedMNIST task '{task}': {labels.shape}")

        image_list = [_to_pil_image(img) for img in images]
        splits[split] = Dataset.from_dict(
            {"image": image_list, "label": labels.tolist()},
            features=features,
        )

    return DatasetDict(splits)

def load_vtab_dataset(
    dataset_name: str,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
) -> DatasetDict:
    dataset_id = resolve_dataset_id(dataset_name)
    return load_dataset(dataset_id, cache_dir=cache_dir, streaming=streaming)


def build_transforms(
    image_processor: AutoImageProcessor,
    train: bool = True,
) -> transforms.Compose:
    size = image_processor.size.get("shortest_edge")
    if size is None:
        size = image_processor.size.get("height", 224)
    mean = getattr(image_processor, "image_mean", None) or IMAGENET_DEFAULT_MEAN
    std = getattr(image_processor, "image_std", None) or IMAGENET_DEFAULT_STD
    normalize = transforms.Normalize(mean=mean, std=std)
    if train:
        return transforms.Compose(
            [
                transforms.Resize(int(size * 1.15), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomResizedCrop(size, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(size + 32, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def preprocess_splits(
    dataset: DatasetDict,
    image_processor: AutoImageProcessor,
    image_column: str = "img",
    label_column: str = "label",
    label_value_to_id: Optional[Dict[Union[int, str], int]] = None,
) -> DatasetDict:
    train_transform = build_transforms(image_processor, train=True)
    eval_transform = build_transforms(image_processor, train=False)

    def _make_transform(transform):
        def _apply(batch):
            images = batch[image_column]
            labels = batch[label_column]
            is_batched = isinstance(images, list)
            if not is_batched:
                images = [images]
                labels = [labels]
            if label_value_to_id is not None:
                labels = [label_value_to_id[label] for label in labels]
            processed = [transform(_to_pil_image(img).convert("RGB")) for img in images]
            if is_batched:
                return {"pixel_values": processed, "labels": labels}
            return {"pixel_values": processed[0], "labels": labels[0]}

        return _apply

    out = DatasetDict()
    for split, ds in dataset.items():
        transform = train_transform if split == "train" else eval_transform
        if isinstance(ds, (Dataset, IterableDataset)):
            out[split] = ds.with_transform(_make_transform(transform))
    return out


def get_label_info(
    dataset: Dataset,
    label_column: str = "label",
) -> Tuple[List[str], Dict[int, str], Dict[str, int], Optional[Dict[Union[int, str], int]]]:
    """
    Returns label names + model mappings.

    - If the dataset exposes ClassLabel.names, those are used (identity mapping).
    - Otherwise, infer unique label values from the dataset and (if needed) return a value->id remap
      so labels become contiguous [0..num_labels-1].
    """
    feature = dataset.features.get(label_column)
    if feature is None and label_column != "label":
        feature = dataset.features.get("label")
    if feature is None and label_column != "labels":
        feature = dataset.features.get("labels")

    if feature is not None and hasattr(feature, "names"):
        names = list(feature.names)
        id2label = {i: name for i, name in enumerate(names)}
        label2id = {name: i for i, name in enumerate(names)}
        return names, id2label, label2id, None

    if isinstance(dataset, IterableDataset):
        raise ValueError(
            "Dataset does not expose ClassLabel names and is streaming/iterable; "
            "please provide class names or use a non-streaming dataset to infer labels."
        )

    try:
        unique_values = dataset.unique(label_column)
    except Exception as exc:  # pragma: no cover
        raise ValueError(
            f"Could not infer label set from column '{label_column}'. "
            "If the dataset does not expose ClassLabel names, please remap labels manually."
        ) from exc

    if not unique_values:
        raise ValueError(f"Label column '{label_column}' appears to be empty.")

    try:
        unique_values = sorted(unique_values)
    except TypeError:
        unique_values = list(unique_values)

    value_to_id: Dict[Union[int, str], int] = {value: idx for idx, value in enumerate(unique_values)}

    if unique_values == list(range(len(unique_values))):
        names = [f"class_{i}" for i in range(len(unique_values))]
        id2label = {i: name for i, name in enumerate(names)}
        label2id = {name: i for i, name in enumerate(names)}
        return names, id2label, label2id, None

    names = [f"class_{value}" for value in unique_values]
    id2label = {i: name for i, name in enumerate(names)}
    label2id = {name: i for i, name in enumerate(names)}
    return names, id2label, label2id, value_to_id

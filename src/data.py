from typing import Dict, List, Optional, Tuple, Union

from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoImageProcessor

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
            processed = [transform(img.convert("RGB")) for img in images]
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

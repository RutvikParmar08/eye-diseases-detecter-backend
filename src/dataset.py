# import os
# def build_datasets(
#     data_dir: str = "data/train",
#     img_size: int = DEFAULT_IMG_SIZE,
#     batch_size: int = DEFAULT_BATCH,
#     val_split: float = DEFAULT_VAL_SPLIT,
#     seed: int = DEFAULT_SEED,
#     ) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str], Dict[int, float]]:
#     """
#     Returns: train_ds, val_ds, class_names, class_weights
#     - Loads from a single root: data/train/<class>/*.jpg
#     - Splits into train/val with a fixed seed for reproducibility
#     """
#     print(f"[DATA] Scanning classes under: {data_dir}")
#     ensure_dir(data_dir)


# class_names = list_class_dirs(data_dir)
# if not class_names:
# raise RuntimeError(
#     f"No classes with images found in '{data_dir}'.\n"
#     f"Expected subfolders like 'cataract', 'diabetic_retinopathy', 'glaucoma', 'normal' with images."
#     )
#     print(f"[DATA] Discovered classes: {class_names}")


# counts = count_images_per_class(data_dir, class_names)
# print(f"[DATA] Image counts per class: {counts}")


# # Sanity: require at least 5 images total and >=1 per class
# if sum(counts.values()) < 5 or any(v == 0 for v in counts.values()):
# raise RuntimeError(
# "Insufficient images. Ensure each class folder contains images (>=1) and total >= 5."
# )


# # Build datasets via image_dataset_from_directory (categorical labels)
# common_kwargs = dict(
# directory=data_dir,
# labels="inferred",
# label_mode="categorical",
# class_names=class_names,
# image_size=(img_size, img_size),
# batch_size=batch_size,
# shuffle=True,
# seed=seed,
# validation_split=val_split,
# )


# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
# subset="training", **common_kwargs
# )
# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
# subset="validation", **common_kwargs
# )


# # Cache/Prefetch for speed
# AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.shuffle(1024, seed=seed).cache().prefetch(AUTOTUNE)
# val_ds = val_ds.cache().prefetch(AUTOTUNE)


# # Compute class weights (for categorical -> index order = class_names order)
# class_weights = compute_class_weights(counts)
# print(f"[DATA] Computed class weights: {class_weights}")


# return train_ds, val_ds, class_names, class_weights
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    return train_loader, train_dataset.classes
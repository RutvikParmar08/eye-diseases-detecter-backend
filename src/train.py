# # import tensorflow as tf
# # from dataset import load_datasets
# # from model import build_model

# # EPOCHS = 20
# # MODEL_PATH = "models/best_model.h5"

# # print("[INFO] Loading datasets...")
# # train_ds, val_ds = load_datasets("data")

# # print("[INFO] Building model...")
# # model = build_model()
# # model.summary()

# # model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='binary_crossentropy',metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# # print("[INFO] Starting training...")
# # callbacks = [
# #     tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_auc', mode='max'),
# #     tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_auc')
# # ]

# # history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# # print("[INFO] Training finished. Model saved to:", MODEL_PATH)
# import os
# import json
# import datetime as dt
# import tensorflow as tf
# from tensorflow.keras import callbacks, optimizers, losses, metrics


# from dataset import build_datasets, save_label_map, ensure_dir
# from model import build_model


# # ---------------- Config ----------------
# DATA_DIR = os.path.join("data", "train")
# MODELS_DIR = "models"
# LOGS_DIR = os.path.join("logs")
# IMG_SIZE = 320
# BATCH_SIZE = 32
# VAL_SPLIT = 0.2
# SEED = 1337
# WARMUP_LR = 1e-3
# FINETUNE_LR = 1e-5
# WARMUP_EPOCHS = 5
# FINETUNE_EPOCHS = 15
# DROPOUT = 0.3


# # --------------- Logging ---------------
# ensure_dir(MODELS_DIR)
# ensure_dir(LOGS_DIR)
# run_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
# ckpt_path = os.path.join(MODELS_DIR, "best.keras")
# hist_path = os.path.join(MODELS_DIR, "history.json")
# labelmap_path = os.path.join(MODELS_DIR, "label_map.json")
# tblog_dir = os.path.join(LOGS_DIR, f"tb_{run_tag}")


# print("[TRAIN] Preparing datasets...")
# train_ds, val_ds, class_names, class_weights = build_datasets(
#     data_dir=DATA_DIR,
#     img_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     val_split=VAL_SPLIT,
#     seed=SEED,
# )


# num_classes = len(class_names)
# print(f"[TRAIN] Classes: {class_names} (num_classes={num_classes})")
# save_label_map(labelmap_path, class_names)


# print("[TRAIN] Building model...")
# model = build_model(num_classes=num_classes, img_size=IMG_SIZE, dropout=DROPOUT)
# model.summary()


# # Choose loss/metrics depending on num_classes
# if num_classes == 2:
#     loss_fn = losses.BinaryCrossentropy()
#     monitor_metric = "val_auc"
#     mets = [metrics.AUC(name="auc"), metrics.BinaryAccuracy(name="acc")]
# else:
# loss_fn = losses.CategoricalCrossentropy()
# monitor_metric = "val_accuracy"
# mets = [metrics.CategoricalAccuracy(name="acc")]


# # Callbacks
# cbs = [
# callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor=monitor_metric, mode="max" if "auc" in monitor_metric or "acc" in monitor_metric else "min"),
# callbacks.ReduceLROnPlateau(monitor=monitor_metric, factor=0.5, patience=2, mode="max" if "auc" in monitor_metric or "acc" in monitor_metric else "min"),
# callbacks.EarlyStopping(monitor=monitor_metric, patience=5, restore_best_weights=True, mode="max" if "auc" in monitor_metric or "acc" in monitor_metric else "min"),
# callbacks.TensorBoard(log_dir=tblog_dir),
# ]


# # -------- Warmup (head only) --------
# print("[TRAIN] Warmup training (frozen backbone)...")
# model.get_layer(index=2).trainable = False # ensure base frozen
# model.compile(optimizer=optimizers.Adam(WARMUP_LR), loss=loss_fn, metrics=mets)
# hist_warm = model.fit(train_ds, validation_data=val_ds, epochs=WARMUP_EPOCHS,
# class_weight=class_weights if class_weights else None,
# callbacks=cbs)


# # -------- Fine-tune (unfreeze top layers) --------
# print("[TRAIN] Fine-tuning top layers...")
# base = None
# for lyr in model.layers:
# if hasattr(lyr, 'name') and 'efficientnetb0' in lyr.name:
# base = lyr
# break
# if base is not None:
# # unfreeze last ~20 layers safely
# for l in base.layers[-20:]:
# l.trainable = True


# model.compile(optimizer=optimizers.Adam(FINETUNE_LR), loss=loss_fn, metrics=mets)
# hist_ft = model.fit(train_ds, validation_data=val_ds, epochs=FINETUNE_EPOCHS,
# class_weight=class_weights if class_weights else None,
# callbacks

import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_dataloaders
from src.model import EyeDiseaseNet


def train_model():
    data_dir = "data"
    train_loader, classes = get_dataloaders(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = EyeDiseaseNet(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    for epoch in range(10):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)


            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader):.4f}")


    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/eye_disease_model.pth")
    print("Model saved!")


if __name__ == "__main__":
    train_model()
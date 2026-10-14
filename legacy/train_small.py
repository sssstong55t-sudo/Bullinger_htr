import os
import random
import warnings
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    logging as transformers_logging,
)

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("/Users/shaotongsmac/Desktop/bullinger-htr")

TRAIN_DIR = (
    ROOT
    / "train"
    / "extracted_0_2000"
    / "0-2000-out"
)

OUTPUT_DIR = ROOT / "trocr-bullinger-small"

MODEL_NAME = "microsoft/trocr-base-handwritten"

NUM_DE = 250
NUM_LA = 250

EPOCHS = 1
BATCH_SIZE = 2
LEARNING_RATE = 5e-5

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print()
print(f"Device: {device}")


# ============================================================
# LOAD PROCESSOR + MODEL
# ============================================================

print("Loading TrOCR...")

processor = TrOCRProcessor.from_pretrained(
    MODEL_NAME
)

model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_NAME
)


# ============================================================
# CONFIGURE DECODER TOKENS
# ============================================================

# Required by VisionEncoderDecoderModel during training.
# Without these values, passing labels to model() can fail
# when the decoder inputs are automatically created.

model.config.decoder_start_token_id = (
    processor.tokenizer.cls_token_id
)

model.config.pad_token_id = (
    processor.tokenizer.pad_token_id
)

model.config.eos_token_id = (
    processor.tokenizer.sep_token_id
)

# Keep generation configuration consistent with model config.
model.generation_config.decoder_start_token_id = (
    processor.tokenizer.cls_token_id
)

model.generation_config.pad_token_id = (
    processor.tokenizer.pad_token_id
)

model.generation_config.eos_token_id = (
    processor.tokenizer.sep_token_id
)


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(device)

model.train()

print("Model loaded.")


# ============================================================
# COLLECT VALID PAIRS
# ============================================================

def collect_pairs(folder):

    pairs = []

    if not folder.exists():
        return pairs

    for img_path in folder.rglob("*.png"):

        txt_path = img_path.with_suffix(".txt")

        if txt_path.exists():

            pairs.append(
                (img_path, txt_path)
            )

    return pairs


de_pairs = collect_pairs(
    TRAIN_DIR / "de"
)

la_pairs = collect_pairs(
    TRAIN_DIR / "la"
)


print()
print(
    f"Available German pairs: "
    f"{len(de_pairs)}"
)

print(
    f"Available Latin pairs:  "
    f"{len(la_pairs)}"
)


# ============================================================
# RANDOM SAMPLE
# ============================================================

random.shuffle(de_pairs)
random.shuffle(la_pairs)

de_pairs = de_pairs[:NUM_DE]
la_pairs = la_pairs[:NUM_LA]

samples = de_pairs + la_pairs

random.shuffle(samples)


print(
    f"Selected German: "
    f"{len(de_pairs)}"
)

print(
    f"Selected Latin:  "
    f"{len(la_pairs)}"
)

print(
    f"Total training samples: "
    f"{len(samples)}"
)


# ============================================================
# DATASET
# ============================================================

class BullingerDataset(Dataset):

    def __init__(
        self,
        samples,
        processor
    ):

        self.samples = samples
        self.processor = processor

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx
    ):

        img_path, txt_path = (
            self.samples[idx]
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            img_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Load ground-truth transcription
        # ----------------------------------------------------

        text = txt_path.read_text(
            encoding="utf-8"
        ).strip()

        # ----------------------------------------------------
        # Image preprocessing
        # ----------------------------------------------------

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)

        # ----------------------------------------------------
        # Text tokenization
        # ----------------------------------------------------

        labels = (
            self.processor.tokenizer(
                text,
                padding="max_length",
                max_length=128,
                truncation=True,
                return_tensors="pt"
            )
            .input_ids
            .squeeze(0)
        )

        # ----------------------------------------------------
        # Padding tokens must not contribute to loss
        # ----------------------------------------------------

        labels[
            labels
            == self.processor.tokenizer.pad_token_id
        ] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


# ============================================================
# CREATE DATASET
# ============================================================

dataset = BullingerDataset(
    samples,
    processor
)


# ============================================================
# DATALOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("START TRAINING")
print("=" * 70)

total_steps = len(loader)


for epoch in range(EPOCHS):

    running_loss = 0.0

    for step, batch in enumerate(
        loader,
        start=1
    ):

        # ----------------------------------------------------
        # Move batch to device
        # ----------------------------------------------------

        pixel_values = batch[
            "pixel_values"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        # ----------------------------------------------------
        # Reset gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(
            pixel_values=pixel_values,
            labels=labels
        )

        loss = outputs.loss

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Update parameters
        # ----------------------------------------------------

        optimizer.step()

        running_loss += loss.item()

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            step % 10 == 0
            or step == total_steps
        ):

            avg_loss = (
                running_loss / step
            )

            print(
                f"Epoch "
                f"{epoch + 1}/{EPOCHS} | "
                f"Step "
                f"{step}/{total_steps} | "
                f"Loss "
                f"{avg_loss:.4f}"
            )


# ============================================================
# SAVE
# ============================================================

print()
print("Training finished.")

print(
    f"Saving model to: "
    f"{OUTPUT_DIR}"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

model.save_pretrained(
    OUTPUT_DIR
)

processor.save_pretrained(
    OUTPUT_DIR
)

print(
    "Model saved successfully."
)

print("=" * 70)
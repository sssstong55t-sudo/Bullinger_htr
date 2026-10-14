import os
import random
import warnings
from pathlib import Path

# ============================================================
# QUIET OUTPUT
# ============================================================

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

DATA_DIR = (
    ROOT
    / "train"
    / "extracted_0_2000"
    / "0-2000-out"
)

MODEL_NAME = "microsoft/trocr-base-handwritten"

OUTPUT_DIR = ROOT / "trocr-overfit-test"

NUM_DE = 25
NUM_LA = 25

EPOCHS = 10

BATCH_SIZE = 2

# Use a smaller LR than the previous smoke test.
LEARNING_RATE = 1e-5

MAX_LENGTH = 128

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
# COLLECT DATA
# ============================================================

def collect_pairs(folder):

    pairs = []

    if not folder.exists():
        return pairs

    for img_path in sorted(folder.rglob("*.png")):

        txt_path = img_path.with_suffix(".txt")

        if txt_path.exists():

            text = txt_path.read_text(
                encoding="utf-8"
            ).strip()

            # Ignore empty transcription
            if text:
                pairs.append(
                    (img_path, txt_path)
                )

    return pairs


de_pairs = collect_pairs(DATA_DIR / "de")
la_pairs = collect_pairs(DATA_DIR / "la")


print()
print(f"German available: {len(de_pairs)}")
print(f"Latin available:  {len(la_pairs)}")


# ============================================================
# SELECT EXACTLY 50 IMAGES
# ============================================================

random.shuffle(de_pairs)
random.shuffle(la_pairs)

selected_de = de_pairs[:NUM_DE]
selected_la = la_pairs[:NUM_LA]

samples = selected_de + selected_la

random.shuffle(samples)


print()
print(f"Selected German: {len(selected_de)}")
print(f"Selected Latin:  {len(selected_la)}")
print(f"Total:           {len(samples)}")


# ============================================================
# LOAD PROCESSOR + MODEL
# ============================================================

print()
print("Loading TrOCR...")

processor = TrOCRProcessor.from_pretrained(
    MODEL_NAME
)

model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_NAME
)


# ============================================================
# TOKEN CONFIG
# ============================================================

tokenizer = processor.tokenizer

decoder_start_token_id = tokenizer.cls_token_id
pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.sep_token_id


if decoder_start_token_id is None:
    raise ValueError(
        "tokenizer.cls_token_id is None"
    )

if pad_token_id is None:
    raise ValueError(
        "tokenizer.pad_token_id is None"
    )

if eos_token_id is None:
    raise ValueError(
        "tokenizer.sep_token_id is None"
    )


model.config.decoder_start_token_id = (
    decoder_start_token_id
)

model.config.pad_token_id = (
    pad_token_id
)

model.config.eos_token_id = (
    eos_token_id
)


model.generation_config.decoder_start_token_id = (
    decoder_start_token_id
)

model.generation_config.pad_token_id = (
    pad_token_id
)

model.generation_config.eos_token_id = (
    eos_token_id
)


print()
print("Token configuration:")
print(
    "decoder_start_token_id:",
    decoder_start_token_id
)
print(
    "pad_token_id:",
    pad_token_id
)
print(
    "eos_token_id:",
    eos_token_id
)


model = model.to(device)


print()
print("Model loaded.")


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

        return len(self.samples)

    def __getitem__(
        self,
        idx
    ):

        img_path, txt_path = self.samples[idx]

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image = Image.open(
            img_path
        ).convert("RGB")

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        text = txt_path.read_text(
            encoding="utf-8"
        ).strip()

        encoded = tokenizer(
            text,
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="pt"
        )

        labels = encoded.input_ids.squeeze(0)

        # ----------------------------------------------------
        # IMPORTANT
        #
        # Ignore padding in cross-entropy loss.
        # ----------------------------------------------------

        labels = labels.clone()

        labels[
            labels == pad_token_id
        ] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "text": text,
            "path": str(img_path)
        }


dataset = BullingerDataset(
    samples,
    processor
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# CER
# ============================================================

def edit_distance(reference, prediction):

    previous = list(
        range(len(prediction) + 1)
    )

    for i, ref_char in enumerate(
        reference,
        start=1
    ):

        current = [i]

        for j, pred_char in enumerate(
            prediction,
            start=1
        ):

            insert_cost = (
                current[j - 1] + 1
            )

            delete_cost = (
                previous[j] + 1
            )

            substitute_cost = (
                previous[j - 1]
                + (ref_char != pred_char)
            )

            current.append(
                min(
                    insert_cost,
                    delete_cost,
                    substitute_cost
                )
            )

        previous = current

    return previous[-1]


def calculate_cer(
    reference,
    prediction
):

    if len(reference) == 0:

        return (
            0.0
            if len(prediction) == 0
            else 1.0
        )

    distance = edit_distance(
        reference,
        prediction
    )

    return distance / len(reference)


# ============================================================
# RECOGNITION
# ============================================================

@torch.no_grad()
def recognize(
    model,
    img_path
):

    image = Image.open(
        img_path
    ).convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values.to(device)

    generated_ids = model.generate(
        pixel_values,
        max_new_tokens=MAX_LENGTH
    )

    prediction = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return prediction.strip()


# ============================================================
# EVALUATE SAME 50 TRAINING IMAGES
# ============================================================

def evaluate_training_set():

    model.eval()

    total_distance = 0
    total_characters = 0

    examples = []

    for index, (
        img_path,
        txt_path
    ) in enumerate(samples):

        reference = txt_path.read_text(
            encoding="utf-8"
        ).strip()

        prediction = recognize(
            model,
            img_path
        )

        distance = edit_distance(
            reference,
            prediction
        )

        total_distance += distance
        total_characters += len(reference)

        # Save first three examples
        if index < 3:

            examples.append(
                (
                    img_path.name,
                    reference,
                    prediction,
                    calculate_cer(
                        reference,
                        prediction
                    )
                )
            )

    if total_characters == 0:
        dataset_cer = 0.0

    else:
        dataset_cer = (
            total_distance
            / total_characters
        )

    return dataset_cer, examples


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# BEFORE TRAINING
# ============================================================

print()
print("=" * 80)
print("BEFORE TRAINING")
print("=" * 80)

before_cer, before_examples = (
    evaluate_training_set()
)

print(
    f"Training-set CER before fine-tuning: "
    f"{before_cer:.4f}"
)


for (
    filename,
    reference,
    prediction,
    sample_cer
) in before_examples:

    print()
    print(filename)

    print("GT:")
    print(reference)

    print("Prediction:")
    print(prediction)

    print(
        f"CER: {sample_cer:.4f}"
    )


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 80)
print("START OVERFIT TEST")
print("=" * 80)


best_cer = float("inf")


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running_loss = 0.0

    for step, batch in enumerate(
        loader,
        start=1
    ):

        pixel_values = batch[
            "pixel_values"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        optimizer.zero_grad()

        outputs = model(
            pixel_values=pixel_values,
            labels=labels
        )

        loss = outputs.loss

        # ----------------------------------------------------
        # Catch invalid loss immediately
        # ----------------------------------------------------

        if not torch.isfinite(loss):

            raise RuntimeError(
                f"Non-finite loss detected: "
                f"{loss.item()}"
            )

        loss.backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item()


    average_loss = (
        running_loss
        / len(loader)
    )


    # ========================================================
    # EVALUATE THE SAME 50 IMAGES
    # ========================================================

    train_cer, examples = (
        evaluate_training_set()
    )


    print()
    print("-" * 80)

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        f"Average loss: {average_loss:.4f}"
    )

    print(
        f"Training CER: {train_cer:.4f}"
    )


    # ========================================================
    # SHOW EXAMPLES
    # ========================================================

    for (
        filename,
        reference,
        prediction,
        sample_cer
    ) in examples:

        print()

        print(filename)

        print("GT:")
        print(reference)

        print("Prediction:")
        print(prediction)

        print(
            f"CER: {sample_cer:.4f}"
        )


    # ========================================================
    # SAVE BEST
    # ========================================================

    if train_cer < best_cer:

        best_cer = train_cer

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

        print()
        print(
            f"New best CER: "
            f"{best_cer:.4f}"
        )

        print(
            f"Saved to: "
            f"{OUTPUT_DIR}"
        )


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 80)
print("OVERFIT TEST FINISHED")
print("=" * 80)

print(
    f"Best training CER: "
    f"{best_cer:.4f}"
)

print(
    f"Best model saved at:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "Interpretation:"
)

print(
    "If CER falls strongly on these SAME 50 images, "
    "the training pipeline can learn the Bullinger data."
)

print(
    "If CER stays very high or predictions collapse, "
    "we need to debug the training/tokenization pipeline "
    "before full training."
)

print("=" * 80)
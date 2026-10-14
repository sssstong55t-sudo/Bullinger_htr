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

ORIGINAL_MODEL = "microsoft/trocr-base-handwritten"

FINETUNED_MODEL = (
    ROOT
    / "trocr-bullinger-small"
)

NUM_TEST = 20

SEED = 12345

random.seed(SEED)


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
# COLLECT IMAGE/TEXT PAIRS
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
    DATA_DIR / "de"
)

la_pairs = collect_pairs(
    DATA_DIR / "la"
)


print()
print(f"German pairs available: {len(de_pairs)}")
print(f"Latin pairs available:  {len(la_pairs)}")


# ============================================================
# IMPORTANT
#
# train_small.py used random seed 42 and selected:
# 250 German + 250 Latin.
#
# Reproduce exactly that selection so we can exclude those
# images from this test.
# ============================================================

random.seed(42)

train_de = de_pairs.copy()
train_la = la_pairs.copy()

random.shuffle(train_de)
random.shuffle(train_la)

train_de = train_de[:250]
train_la = train_la[:250]


trained_images = {
    img_path
    for img_path, _ in train_de + train_la
}


# ============================================================
# BUILD UNSEEN TEST SET
# ============================================================

unseen_de = [
    pair
    for pair in de_pairs
    if pair[0] not in trained_images
]

unseen_la = [
    pair
    for pair in la_pairs
    if pair[0] not in trained_images
]


print()
print(f"Unseen German: {len(unseen_de)}")
print(f"Unseen Latin:  {len(unseen_la)}")


# ============================================================
# SELECT 10 GERMAN + 10 LATIN
# ============================================================

random.seed(SEED)

num_de = NUM_TEST // 2
num_la = NUM_TEST - num_de

test_de = random.sample(
    unseen_de,
    min(num_de, len(unseen_de))
)

test_la = random.sample(
    unseen_la,
    min(num_la, len(unseen_la))
)

test_samples = test_de + test_la

random.shuffle(test_samples)


print()
print(f"Testing {len(test_samples)} unseen images.")


# ============================================================
# LOAD PROCESSOR
# ============================================================

print()
print("Loading processor...")

processor = TrOCRProcessor.from_pretrained(
    ORIGINAL_MODEL
)


# ============================================================
# LOAD ORIGINAL MODEL
# ============================================================

print("Loading ORIGINAL TrOCR...")

original_model = (
    VisionEncoderDecoderModel
    .from_pretrained(
        ORIGINAL_MODEL
    )
    .to(device)
)

original_model.eval()


# ============================================================
# LOAD FINE-TUNED MODEL
# ============================================================

print("Loading FINE-TUNED model...")

finetuned_model = (
    VisionEncoderDecoderModel
    .from_pretrained(
        FINETUNED_MODEL
    )
    .to(device)
)

finetuned_model.eval()


print("Models loaded.")


# ============================================================
# RECOGNITION FUNCTION
# ============================================================

@torch.no_grad()
def recognize(model, image):

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values.to(device)

    generated_ids = model.generate(
        pixel_values,
        max_new_tokens=128
    )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return text.strip()


# ============================================================
# SIMPLE CER
# ============================================================

def edit_distance(a, b):

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):

        current = [i]

        for j, char_b in enumerate(b, start=1):

            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1

            replace_cost = (
                previous[j - 1]
                + (char_a != char_b)
            )

            current.append(
                min(
                    insert_cost,
                    delete_cost,
                    replace_cost
                )
            )

        previous = current

    return previous[-1]


def cer(reference, prediction):

    if len(reference) == 0:

        if len(prediction) == 0:
            return 0.0

        return 1.0

    distance = edit_distance(
        reference,
        prediction
    )

    return distance / len(reference)


# ============================================================
# TEST
# ============================================================

print()
print("=" * 80)
print("START COMPARISON")
print("=" * 80)


original_cers = []
finetuned_cers = []


for index, (img_path, txt_path) in enumerate(
    test_samples,
    start=1
):

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ground_truth = txt_path.read_text(
        encoding="utf-8"
    ).strip()

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    image = Image.open(
        img_path
    ).convert("RGB")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    original_prediction = recognize(
        original_model,
        image
    )

    finetuned_prediction = recognize(
        finetuned_model,
        image
    )

    # --------------------------------------------------------
    # CER
    # --------------------------------------------------------

    original_cer = cer(
        ground_truth,
        original_prediction
    )

    finetuned_cer = cer(
        ground_truth,
        finetuned_prediction
    )

    original_cers.append(
        original_cer
    )

    finetuned_cers.append(
        finetuned_cer
    )

    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    if "/de/" in str(img_path):
        language = "DE"
    else:
        language = "LA"

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("-" * 80)

    print(
        f"[{index}/{len(test_samples)}] "
        f"Language: {language}"
    )

    print(
        f"Image: {img_path.name}"
    )

    print()

    print("GROUND TRUTH:")
    print(ground_truth)

    print()

    print(
        f"ORIGINAL "
        f"(CER={original_cer:.3f}):"
    )

    print(original_prediction)

    print()

    print(
        f"FINE-TUNED "
        f"(CER={finetuned_cer:.3f}):"
    )

    print(finetuned_prediction)


# ============================================================
# SUMMARY
# ============================================================

avg_original_cer = (
    sum(original_cers)
    / len(original_cers)
)

avg_finetuned_cer = (
    sum(finetuned_cers)
    / len(finetuned_cers)
)


print()
print("=" * 80)
print("RESULT")
print("=" * 80)

print(
    f"Original TrOCR average CER:   "
    f"{avg_original_cer:.4f}"
)

print(
    f"Fine-tuned model average CER: "
    f"{avg_finetuned_cer:.4f}"
)


difference = (
    avg_original_cer
    - avg_finetuned_cer
)


print()

if difference > 0:

    print(
        "Fine-tuned model improved CER by: "
        f"{difference:.4f}"
    )

elif difference < 0:

    print(
        "Fine-tuned model CER became worse by: "
        f"{abs(difference):.4f}"
    )

else:

    print(
        "Both models have the same CER."
    )


print()
print("Comparison finished.")
print("=" * 80)
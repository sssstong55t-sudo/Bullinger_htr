import os
import random
import warnings
import json
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

OUTPUT_DIR = ROOT / "trocr-bullinger-medium"

SPLIT_FILE = ROOT / "medium_split.json"

TRAIN_SIZE = 4000
VAL_SIZE = 500
TEST_SIZE = 500

EPOCHS = 5

BATCH_SIZE = 2

LEARNING_RATE = 1e-5

MAX_LENGTH = 128

SEED = 42

NUM_WORKERS = 0

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
# COLLECT IMAGE/TEXT PAIRS
# ============================================================

def collect_pairs(folder, language):

    pairs = []

    if not folder.exists():
        return pairs

    for img_path in sorted(folder.rglob("*.png")):

        txt_path = img_path.with_suffix(".txt")

        if not txt_path.exists():
            continue

        text = txt_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            continue

        pairs.append({
            "image": str(img_path),
            "text": str(txt_path),
            "language": language,
        })

    return pairs


print()
print("Collecting dataset...")

de_pairs = collect_pairs(
    DATA_DIR / "de",
    "de"
)

la_pairs = collect_pairs(
    DATA_DIR / "la",
    "la"
)


print(f"German available: {len(de_pairs)}")
print(f"Latin available:  {len(la_pairs)}")
print(
    f"Total available:  "
    f"{len(de_pairs) + len(la_pairs)}"
)


# ============================================================
# STRATIFIED SPLIT
# ============================================================

def stratified_take(
    de_data,
    la_data,
    total_size,
    rng
):

    total_available = (
        len(de_data)
        + len(la_data)
    )

    de_ratio = (
        len(de_data)
        / total_available
    )

    num_de = round(
        total_size * de_ratio
    )

    num_la = (
        total_size - num_de
    )

    selected_de = de_data[:num_de]
    selected_la = la_data[:num_la]

    del de_data[:num_de]
    del la_data[:num_la]

    selected = (
        selected_de
        + selected_la
    )

    rng.shuffle(selected)

    return selected


# ============================================================
# CREATE OR LOAD FIXED SPLIT
# ============================================================

if SPLIT_FILE.exists():

    print()
    print(
        f"Loading existing split:"
    )

    print(SPLIT_FILE)

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        split_data = json.load(f)

    train_samples = split_data["train"]
    val_samples = split_data["validation"]
    test_samples = split_data["test"]


else:

    print()
    print("Creating fixed split...")

    rng = random.Random(SEED)

    de_pool = de_pairs.copy()
    la_pool = la_pairs.copy()

    rng.shuffle(de_pool)
    rng.shuffle(la_pool)

    train_samples = stratified_take(
        de_pool,
        la_pool,
        TRAIN_SIZE,
        rng
    )

    val_samples = stratified_take(
        de_pool,
        la_pool,
        VAL_SIZE,
        rng
    )

    test_samples = stratified_take(
        de_pool,
        la_pool,
        TEST_SIZE,
        rng
    )

    split_data = {
        "seed": SEED,
        "train": train_samples,
        "validation": val_samples,
        "test": test_samples,
    }

    with open(
        SPLIT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            split_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Split saved to:"
    )

    print(SPLIT_FILE)


# ============================================================
# SPLIT SUMMARY
# ============================================================

def count_languages(samples):

    de = sum(
        1
        for x in samples
        if x["language"] == "de"
    )

    la = sum(
        1
        for x in samples
        if x["language"] == "la"
    )

    return de, la


train_de, train_la = count_languages(
    train_samples
)

val_de, val_la = count_languages(
    val_samples
)

test_de, test_la = count_languages(
    test_samples
)


print()
print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    f"TRAIN: {len(train_samples)} "
    f"(DE={train_de}, LA={train_la})"
)

print(
    f"VAL:   {len(val_samples)} "
    f"(DE={val_de}, LA={val_la})"
)

print(
    f"TEST:  {len(test_samples)} "
    f"(DE={test_de}, LA={test_la})"
)


# ============================================================
# CHECK SPLIT OVERLAP
# ============================================================

train_paths = {
    x["image"]
    for x in train_samples
}

val_paths = {
    x["image"]
    for x in val_samples
}

test_paths = {
    x["image"]
    for x in test_samples
}


assert train_paths.isdisjoint(
    val_paths
)

assert train_paths.isdisjoint(
    test_paths
)

assert val_paths.isdisjoint(
    test_paths
)


print()
print("Split overlap check: OK")


# ============================================================
# LOAD PROCESSOR + MODEL
# ============================================================

print()
print("Loading TrOCR...")

processor = TrOCRProcessor.from_pretrained(
    MODEL_NAME
)

tokenizer = processor.tokenizer

model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_NAME
)


# ============================================================
# TOKEN CONFIGURATION
# ============================================================

decoder_start_token_id = (
    tokenizer.cls_token_id
)

pad_token_id = (
    tokenizer.pad_token_id
)

eos_token_id = (
    tokenizer.sep_token_id
)


if decoder_start_token_id is None:
    raise ValueError(
        "decoder_start_token_id is None"
    )

if pad_token_id is None:
    raise ValueError(
        "pad_token_id is None"
    )

if eos_token_id is None:
    raise ValueError(
        "eos_token_id is None"
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


model = model.to(device)


print("Model loaded.")

print(
    f"decoder_start_token_id: "
    f"{decoder_start_token_id}"
)

print(
    f"pad_token_id: "
    f"{pad_token_id}"
)

print(
    f"eos_token_id: "
    f"{eos_token_id}"
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

        sample = self.samples[idx]

        img_path = Path(
            sample["image"]
        )

        txt_path = Path(
            sample["text"]
        )

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

        labels = tokenizer(
            text,
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        labels = labels.clone()

        labels[
            labels == pad_token_id
        ] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = BullingerDataset(
    train_samples,
    processor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)


# ============================================================
# EDIT DISTANCE / CER
# ============================================================

def edit_distance(
    reference,
    prediction
):

    previous = list(
        range(
            len(prediction) + 1
        )
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
                + (
                    ref_char
                    != pred_char
                )
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


# ============================================================
# RECOGNIZE ONE IMAGE
# ============================================================

@torch.no_grad()
def recognize(
    current_model,
    img_path
):

    image = Image.open(
        img_path
    ).convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values.to(device)

    generated_ids = (
        current_model.generate(
            pixel_values,
            max_new_tokens=MAX_LENGTH
        )
    )

    prediction = (
        processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        .strip()
    )

    return prediction


# ============================================================
# EVALUATE CER
#
# Uses corpus-level CER:
#
# total edit distance
# -------------------
# total reference chars
# ============================================================

def evaluate_cer(
    current_model,
    samples,
    name
):

    current_model.eval()

    total_distance = 0
    total_chars = 0

    de_distance = 0
    de_chars = 0

    la_distance = 0
    la_chars = 0


    print()

    print(
        f"Evaluating {name}: "
        f"{len(samples)} samples"
    )


    for index, sample in enumerate(
        samples,
        start=1
    ):

        img_path = Path(
            sample["image"]
        )

        txt_path = Path(
            sample["text"]
        )

        language = sample[
            "language"
        ]

        reference = (
            txt_path.read_text(
                encoding="utf-8"
            )
            .strip()
        )

        prediction = recognize(
            current_model,
            img_path
        )

        distance = edit_distance(
            reference,
            prediction
        )

        chars = len(reference)

        total_distance += distance
        total_chars += chars


        if language == "de":

            de_distance += distance
            de_chars += chars

        else:

            la_distance += distance
            la_chars += chars


        if (
            index % 50 == 0
            or index == len(samples)
        ):

            print(
                f"{name}: "
                f"{index}/{len(samples)}"
            )


    overall_cer = (
        total_distance / total_chars
        if total_chars
        else 0.0
    )

    de_cer = (
        de_distance / de_chars
        if de_chars
        else 0.0
    )

    la_cer = (
        la_distance / la_chars
        if la_chars
        else 0.0
    )


    return {
        "overall": overall_cer,
        "de": de_cer,
        "la": la_cer,
    }


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# BASELINE VALIDATION CER
# ============================================================

print()
print("=" * 70)
print("ORIGINAL MODEL VALIDATION BASELINE")
print("=" * 70)

baseline_val = evaluate_cer(
    model,
    val_samples,
    "Original validation"
)


print()
print(
    f"Original Validation CER: "
    f"{baseline_val['overall']:.4f}"
)

print(
    f"  German CER: "
    f"{baseline_val['de']:.4f}"
)

print(
    f"  Latin CER:  "
    f"{baseline_val['la']:.4f}"
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("START TRAINING")
print("=" * 70)

best_val_cer = float("inf")

history = []


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running_loss = 0.0

    total_steps = len(
        train_loader
    )


    print()
    print("=" * 70)

    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )

    print("=" * 70)


    for step, batch in enumerate(
        train_loader,
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


        if not torch.isfinite(loss):

            raise RuntimeError(
                f"Non-finite loss: "
                f"{loss.item()}"
            )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


        running_loss += (
            loss.item()
        )


        if (
            step % 100 == 0
            or step == total_steps
        ):

            current_avg = (
                running_loss / step
            )

            print(
                f"Training: "
                f"{step}/{total_steps} | "
                f"Avg Loss: "
                f"{current_avg:.4f}"
            )


    # ========================================================
    # TRAIN LOSS
    # ========================================================

    train_loss = (
        running_loss
        / total_steps
    )


    # ========================================================
    # VALIDATION CER
    # ========================================================

    val_result = evaluate_cer(
        model,
        val_samples,
        f"Validation epoch {epoch}"
    )


    val_cer = (
        val_result["overall"]
    )


    print()
    print("-" * 70)

    print(
        f"Epoch {epoch} finished"
    )

    print(
        f"Train Loss:     "
        f"{train_loss:.4f}"
    )

    print(
        f"Validation CER: "
        f"{val_cer:.4f}"
    )

    print(
        f"  German CER:   "
        f"{val_result['de']:.4f}"
    )

    print(
        f"  Latin CER:    "
        f"{val_result['la']:.4f}"
    )

    print("-" * 70)


    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_cer": val_cer,
        "validation_de_cer": (
            val_result["de"]
        ),
        "validation_la_cer": (
            val_result["la"]
        ),
    })


    # ========================================================
    # SAVE BEST CHECKPOINT
    # ========================================================

    if val_cer < best_val_cer:

        best_val_cer = val_cer


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
            "★ NEW BEST MODEL"
        )

        print(
            f"Validation CER: "
            f"{best_val_cer:.4f}"
        )

        print(
            f"Saved to:"
        )

        print(
            OUTPUT_DIR
        )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_file = (
        OUTPUT_DIR
        / "training_history.json"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        history_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# RELEASE CURRENT MODEL BEFORE LOADING TWO MODELS
# ============================================================

del model

if device.type == "mps":
    torch.mps.empty_cache()

elif device.type == "cuda":
    torch.cuda.empty_cache()


# ============================================================
# LOAD ORIGINAL MODEL FOR FINAL TEST
# ============================================================

print()
print("=" * 70)
print("FINAL TEST")
print("=" * 70)

print()
print(
    "Loading ORIGINAL model..."
)

original_model = (
    VisionEncoderDecoderModel
    .from_pretrained(
        MODEL_NAME
    )
)

original_model.config.decoder_start_token_id = (
    decoder_start_token_id
)

original_model.config.pad_token_id = (
    pad_token_id
)

original_model.config.eos_token_id = (
    eos_token_id
)

original_model.generation_config.decoder_start_token_id = (
    decoder_start_token_id
)

original_model.generation_config.pad_token_id = (
    pad_token_id
)

original_model.generation_config.eos_token_id = (
    eos_token_id
)

original_model = (
    original_model.to(device)
)

original_model.eval()


# ============================================================
# ORIGINAL TEST CER
# ============================================================

original_test = evaluate_cer(
    original_model,
    test_samples,
    "Original test"
)


print()
print(
    f"Original Test CER: "
    f"{original_test['overall']:.4f}"
)

print(
    f"  German CER: "
    f"{original_test['de']:.4f}"
)

print(
    f"  Latin CER:  "
    f"{original_test['la']:.4f}"
)


# ============================================================
# RELEASE ORIGINAL MODEL
# ============================================================

del original_model

if device.type == "mps":
    torch.mps.empty_cache()

elif device.type == "cuda":
    torch.cuda.empty_cache()


# ============================================================
# LOAD BEST FINE-TUNED MODEL
# ============================================================

print()
print(
    "Loading BEST fine-tuned model..."
)

best_model = (
    VisionEncoderDecoderModel
    .from_pretrained(
        OUTPUT_DIR
    )
    .to(device)
)

best_model.eval()


# ============================================================
# BEST MODEL TEST CER
# ============================================================

finetuned_test = evaluate_cer(
    best_model,
    test_samples,
    "Fine-tuned test"
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print()

print("ORIGINAL TrOCR")

print(
    f"Overall CER: "
    f"{original_test['overall']:.4f}"
)

print(
    f"German CER:  "
    f"{original_test['de']:.4f}"
)

print(
    f"Latin CER:   "
    f"{original_test['la']:.4f}"
)


print()

print("FINE-TUNED TrOCR")

print(
    f"Overall CER: "
    f"{finetuned_test['overall']:.4f}"
)

print(
    f"German CER:  "
    f"{finetuned_test['de']:.4f}"
)

print(
    f"Latin CER:   "
    f"{finetuned_test['la']:.4f}"
)


improvement = (
    original_test["overall"]
    - finetuned_test["overall"]
)


print()

if improvement > 0:

    print(
        f"CER improvement: "
        f"{improvement:.4f}"
    )

    relative_improvement = (
        improvement
        / original_test["overall"]
        * 100
    )

    print(
        f"Relative CER reduction: "
        f"{relative_improvement:.2f}%"
    )

else:

    print(
        f"CER became worse by: "
        f"{abs(improvement):.4f}"
    )


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

results = {

    "config": {
        "train_size": TRAIN_SIZE,
        "validation_size": VAL_SIZE,
        "test_size": TEST_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
    },

    "best_validation_cer": (
        best_val_cer
    ),

    "original_test": (
        original_test
    ),

    "finetuned_test": (
        finetuned_test
    ),

    "absolute_cer_improvement": (
        improvement
    ),
}


results_file = (
    OUTPUT_DIR
    / "final_results.json"
)


with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print(
    f"Results saved to:"
)

print(
    results_file
)

print()
print(
    f"Best model:"
)

print(
    OUTPUT_DIR
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
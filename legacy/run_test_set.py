import os
import warnings
from pathlib import Path

# ============================================================
# 安静模式：隐藏 HuggingFace / Transformers 的无关输出
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
from jiwer import cer, wer

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()


# ============================================================
# 配置
# ============================================================

ROOT = Path("/Users/shaotongsmac/Desktop/bullinger-htr")
TEST_DIR = ROOT / "test"
TEST_FILES = ROOT / "test.files"

MODEL_NAME = "microsoft/trocr-base-handwritten"

NUM_DE = 50
NUM_LA = 50


# ============================================================
# 设备
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"\nDevice: {device}")


# ============================================================
# 加载模型
# ============================================================

print("Loading TrOCR...")

processor = TrOCRProcessor.from_pretrained(MODEL_NAME)

model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_NAME
).to(device)

model.eval()

print("Model loaded.\n")


# ============================================================
# 读取官方 test split
# ============================================================

with open(TEST_FILES, "r", encoding="utf-8") as f:
    test_paths = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(f"Official test set: {len(test_paths)} samples")


# ============================================================
# 根据语言选择样本
# ============================================================

de_samples = []
la_samples = []

for relative_txt in test_paths:

    txt_path = TEST_DIR / relative_txt
    img_path = txt_path.with_suffix(".png")

    if not txt_path.exists() or not img_path.exists():
        continue

    parts = Path(relative_txt).parts

    if "de" in parts and len(de_samples) < NUM_DE:
        de_samples.append((img_path, txt_path))

    elif "la" in parts and len(la_samples) < NUM_LA:
        la_samples.append((img_path, txt_path))

    if len(de_samples) >= NUM_DE and len(la_samples) >= NUM_LA:
        break


print(f"German samples: {len(de_samples)}")
print(f"Latin samples:  {len(la_samples)}")


# ============================================================
# 单个语言测试函数
# ============================================================

def evaluate(samples, language):

    print("\n" + "=" * 80)
    print(f"{language} TEST")
    print("=" * 80)

    ground_truths = []
    predictions = []

    for i, (img_path, txt_path) in enumerate(samples, start=1):

        gt = txt_path.read_text(
            encoding="utf-8"
        ).strip()

        image = Image.open(
            img_path
        ).convert("RGB")

        pixel_values = processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=128
            )

        pred = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        ground_truths.append(gt)
        predictions.append(pred)

        # 为了不刷屏，只显示每 10 个样本的进度
        if i % 10 == 0 or i == len(samples):
            print(f"Processed {i}/{len(samples)}")

    language_cer = cer(
        ground_truths,
        predictions
    )

    language_wer = wer(
        ground_truths,
        predictions
    )

    print("\nResults")
    print("-" * 40)
    print(f"CER: {language_cer * 100:.2f}%")
    print(f"WER: {language_wer * 100:.2f}%")

    return (
        ground_truths,
        predictions,
        language_cer,
        language_wer
    )


# ============================================================
# German
# ============================================================

de_gt, de_pred, de_cer, de_wer = evaluate(
    de_samples,
    "GERMAN (de)"
)


# ============================================================
# Latin
# ============================================================

la_gt, la_pred, la_cer, la_wer = evaluate(
    la_samples,
    "LATIN (la)"
)


# ============================================================
# Overall
# ============================================================

all_gt = de_gt + la_gt
all_pred = de_pred + la_pred

overall_cer = cer(
    all_gt,
    all_pred
)

overall_wer = wer(
    all_gt,
    all_pred
)


# ============================================================
# 最终结果
# ============================================================

print("\n")
print("=" * 80)
print("FINAL RESULTS")
print("=" * 80)

print(f"""
German ({len(de_gt)} samples)
CER: {de_cer * 100:.2f}%
WER: {de_wer * 100:.2f}%

Latin ({len(la_gt)} samples)
CER: {la_cer * 100:.2f}%
WER: {la_wer * 100:.2f}%

Overall ({len(all_gt)} samples)
CER: {overall_cer * 100:.2f}%
WER: {overall_wer * 100:.2f}%
""")

print("=" * 80)
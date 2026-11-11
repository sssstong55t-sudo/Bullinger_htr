"""Decode the evaluated mBERT checkpoint into literal character spans."""
import torch

LABELS = ['O', 'B-LOC', 'I-LOC']
INPUT_FIELDS = ('input_ids', 'attention_mask', 'token_type_ids')


def windows(rows, tokenizer):
    features = []
    for row_index, row in enumerate(rows):
        encoded = tokenizer(
            row['text'], truncation=True, max_length=128, stride=32,
            return_overflowing_tokens=True, return_offsets_mapping=True,
            return_token_type_ids=True,
        )
        for index, offsets in enumerate(encoded['offset_mapping']):
            feature = {name: encoded[name][index] for name in INPUT_FIELDS}
            features.append(dict(feature, row_index=row_index, offsets=offsets))
    return features


def collate(features, pad_token_id):
    width = max(len(feature['input_ids']) for feature in features)
    batch = {}
    for name in INPUT_FIELDS:
        pad = pad_token_id if name == 'input_ids' else 0
        batch[name] = torch.tensor([
            feature[name] + [pad] * (width - len(feature[name]))
            for feature in features
        ], dtype=torch.long)
    return batch


def decode(text, tokens):
    spans = []
    current = None

    def finish():
        if current is not None:
            start, end, scores = current
            spans.append(dict(
                start=start, end=end, text=text[start:end],
                score=sum(scores) / len(scores),
            ))

    for (start, end), (_, label, score) in sorted(tokens.items()):
        if label == 0:
            finish()
            current = None
        elif label == 1 or current is None:
            finish()
            current = [start, end, [score]]
        else:
            current[1] = end
            current[2].append(score)
    finish()
    return spans


def predict(model, tokenizer, rows, batch_size=8):
    features = windows(rows, tokenizer)
    tokens = [{} for _ in rows]
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            chunk = features[start:start + batch_size]
            batch = collate(chunk, tokenizer.pad_token_id)
            probabilities = model(**batch).logits.softmax(-1).cpu()
            for index, feature in enumerate(chunk):
                row_tokens = tokens[feature['row_index']]
                for position, (left, right) in enumerate(feature['offsets']):
                    if right <= left:
                        continue
                    # Favor the window with the most context around this token.
                    quality = min(position, len(feature['offsets']) - 1 - position)
                    existing = row_tokens.get((left, right))
                    if existing is None or quality > existing[0]:
                        score, label = probabilities[index, position].max(dim=-1)
                        row_tokens[left, right] = (quality, int(label), float(score))
    return [
        dict(mentions=decode(row['text'], row_tokens))
        for row, row_tokens in zip(rows, tokens)
    ]

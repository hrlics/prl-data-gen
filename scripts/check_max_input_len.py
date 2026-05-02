from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

DATASET = "haoranli-ml/genvf-multi-policy-train-v1_final_bulle_list_final_filtered_threshold_0.6"
MODEL = "Qwen/Qwen3-8B"
COLUMN = "input_to_VF"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
ds = load_dataset(DATASET)

for split, data in ds.items():
    if COLUMN not in data.column_names:
        print(f"[{split}] column '{COLUMN}' not found. Columns: {data.column_names}")
        continue

    max_len = 0
    max_idx = -1
    total = 0
    lengths = []
    for i, text in enumerate(tqdm(data[COLUMN], desc=f"tokenizing {split}")):
        if text is None:
            continue
        n = len(tokenizer.encode(text, add_special_tokens=False))
        lengths.append(n)
        total += n
        if n > max_len:
            max_len = n
            max_idx = i

    lengths.sort()
    count = len(lengths)
    p50 = lengths[count // 2]
    p95 = lengths[int(count * 0.95)]
    p99 = lengths[int(count * 0.99)]
    print(f"\n[{split}] rows={count}  max={max_len} (idx={max_idx})  "
          f"mean={total / count:.1f}  p50={p50}  p95={p95}  p99={p99}")

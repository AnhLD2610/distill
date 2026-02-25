import pandas as pd
from transformers import AutoTokenizer

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "Qwen/Qwen3-14B"
INPUT_PATH = "/home/ubuntu/Working/neurips/train.parquet"
N_SHOW = 3          # số sample muốn xem
SHOW_CHARS = 5000   # cắt bớt cho đỡ dài khi print

# ============================================================
# LOAD
# ============================================================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading dataset...")
df = pd.read_parquet(INPUT_PATH)
print(f"Total rows: {len(df)}")

# ============================================================
# BUILD PROMPT (apply_chat_template)
# ============================================================
def build_prompt(row):
    """
    row["prompt"] thường là numpy array / list các dict: [{"role":..., "content":...}, ...]
    Convert sang list rồi apply chat template.
    """
    messages = row["prompt"]
    if hasattr(messages, "tolist"):  # numpy array
        messages = messages.tolist()
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,   # giữ đúng như bạn đang dùng
    )
    return text

# ============================================================
# PREVIEW
# ============================================================
print("\n================ PROMPT PREVIEW ================")
for i in range(min(N_SHOW, len(df))):
    prompt_text = build_prompt(df.iloc[i])

    print(f"\n--- Row {i} ---")
    # nếu bạn muốn xem luôn raw messages:
    # print("Messages:", df.iloc[i]["prompt"])

    print(prompt_text[:SHOW_CHARS])
    if len(prompt_text) > SHOW_CHARS:
        print(f"\n...[TRUNCATED {len(prompt_text) - SHOW_CHARS} chars]...")

print("\nDone ✅")
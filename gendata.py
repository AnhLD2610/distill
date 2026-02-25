import pandas as pd
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "Qwen/Qwen3-14B"
INPUT_PATH = "/home/ubuntu/Working/neurips/train.parquet"
OUTPUT_PATH = "/home/ubuntu/Working/neurips/train_with_output.parquet"
MAX_TOKENS = 16384
MAX_RETRIES = 10  # số lần retry nếu gen hết token mà chưa end

# ============================================================
# LOAD
# ============================================================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading vLLM model...")
llm = LLM(
    model=MODEL_NAME,
    dtype="auto",
    max_model_len=32768,
    tensor_parallel_size=2,  # chỉnh theo số GPU
    gpu_memory_utilization=0.90,   
)

sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=0.6,
    top_p=0.95,
    stop_token_ids=[151645, 151643],  # <|im_end|>, <|endoftext|>
)

# ============================================================
# LOAD DATA
# ============================================================
print("Loading dataset...")
df = pd.read_parquet(INPUT_PATH)
print(f"Total rows: {len(df)}")

# ============================================================
# BUILD PROMPTS (apply_chat_template)
# ============================================================
def build_prompt(row):
    """Extract messages from prompt field and apply chat template."""
    messages = row["prompt"].tolist()  # numpy array of dicts
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return text

print("Building prompts...")
prompts = [build_prompt(row) for _, row in df.iterrows()]

# ============================================================
# GENERATE WITH RETRY (nếu finish_reason == "length" thì gen lại)
# ============================================================
def generate_batch(prompts_list, indices_to_gen, current_outputs):
    """
    Generate for a list of (original_index, prompt).
    Returns dict: {original_index: output_text}
    """
    batch_prompts = [prompts_list[i] for i in indices_to_gen]
    outputs = llm.generate(batch_prompts, sampling_params)

    results = {}
    retry_indices = []

    for idx, output in zip(indices_to_gen, outputs):
        finish_reason = output.outputs[0].finish_reason
        text = output.outputs[0].text

        if finish_reason == "length":
            # Chưa end, cần retry
            retry_indices.append(idx)
            print(f"  ⚠️  Row {idx} hit max_tokens ({MAX_TOKENS}), will retry...")
        else:
            results[idx] = text

    return results, retry_indices


print("Generating outputs...")
all_results = {}
indices_to_gen = list(range(len(df)))

for attempt in range(1, MAX_RETRIES + 1):
    if not indices_to_gen:
        break
    print(f"\n[Attempt {attempt}] Generating {len(indices_to_gen)} samples...")
    results, retry_indices = generate_batch(prompts, indices_to_gen, all_results)
    all_results.update(results)
    indices_to_gen = retry_indices

# Nếu sau MAX_RETRIES vẫn còn hit length, lưu lại raw output
if indices_to_gen:
    print(f"\n⚠️  {len(indices_to_gen)} rows still hit max_tokens after {MAX_RETRIES} retries. Saving raw output anyway.")
    # Gen lần cuối, lưu dù chưa end
    batch_prompts = [prompts[i] for i in indices_to_gen]
    outputs = llm.generate(batch_prompts, sampling_params)
    for idx, output in zip(indices_to_gen, outputs):
        all_results[idx] = output.outputs[0].text + "  [TRUNCATED]"

# ============================================================
# PARSE THINKING vs CONTENT
# ============================================================
THINK_END_TOKEN_ID = 151668  # </think>

def parse_output(text):
    """Tách thinking content và final content."""
    # Encode lại để tìm </think> token
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    try:
        index = len(token_ids) - token_ids[::-1].index(THINK_END_TOKEN_ID)
        thinking = tokenizer.decode(token_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(token_ids[index:], skip_special_tokens=True).strip("\n")
    except ValueError:
        thinking = ""
        content = text.strip("\n")
    return thinking, content


print("\nParsing thinking content...")
thinking_list = []
content_list = []

for i in range(len(df)):
    raw = all_results.get(i, "")
    thinking, content = parse_output(raw)
    thinking_list.append(thinking)
    content_list.append(content)

df["raw_output"] = [all_results.get(i, "") for i in range(len(df))]
df["thinking_content"] = thinking_list
df["content"] = content_list

# ============================================================
# SAVE
# ============================================================
print(f"\nSaving to {OUTPUT_PATH}...")
df.to_parquet(OUTPUT_PATH, index=False)
print("Done! ✅")

# Quick sanity check
print("\n--- Sample output (row 0) ---")
print("Teacher:", df.iloc[0]["extra_info"].get("opd_teacher"))
print("Thinking:", df.iloc[0]["thinking_content"][:200], "...")
print("Content:", df.iloc[0]["content"][:200], "...")
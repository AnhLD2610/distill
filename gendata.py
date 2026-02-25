import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "Qwen/Qwen3-14B"
MATH_PATH = "/home/ubuntu/Working/neurips/train_math.parquet"
CODE_PATH = "/home/ubuntu/Working/neurips/train_code.parquet"
OUTPUT_PATH = "/home/ubuntu/Working/neurips/train_with_output.parquet"
MAX_TOKENS = 16384
MAX_RETRIES = 10

# ============================================================
# LOAD & MERGE 2 FILES
# ============================================================
print("Loading datasets...")
df_math = pd.read_parquet(MATH_PATH)
df_code = pd.read_parquet(CODE_PATH)
print(f"Math rows: {len(df_math)}, Code rows: {len(df_code)}")

df = pd.concat([df_math, df_code], ignore_index=True)  # ignore_index để reset index 0..N
print(f"Total rows after merge: {len(df)}")

# ============================================================
# LOAD MODEL
# ============================================================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading vLLM model...")
llm = LLM(
    model=MODEL_NAME,
    dtype="auto",
    max_model_len=32768,
    tensor_parallel_size=8,
    gpu_memory_utilization=0.90,
)

sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=0.6,
    top_p=0.95,
    stop_token_ids=[151645, 151643],
)

# ============================================================
# BUILD PROMPTS
# ============================================================
def build_prompt(row):
    messages = row["prompt"].tolist()
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

print("Building prompts...")
prompts = [build_prompt(row) for _, row in df.iterrows()]

# ============================================================
# GENERATE WITH RETRY
# ============================================================
def generate_batch(indices_to_gen):
    batch_prompts = [prompts[i] for i in indices_to_gen]
    outputs = llm.generate(batch_prompts, sampling_params)

    results = {}
    retry_indices = []

    for idx, output in zip(indices_to_gen, outputs):
        if output.outputs[0].finish_reason == "length":
            retry_indices.append(idx)
            print(f"  ⚠️  Row {idx} hit max_tokens, will retry...")
        else:
            results[idx] = output.outputs[0].text

    return results, retry_indices


print("Generating outputs...")
all_results = {}
indices_to_gen = list(range(len(df)))

for attempt in range(1, MAX_RETRIES + 1):
    if not indices_to_gen:
        break
    print(f"\n[Attempt {attempt}] Generating {len(indices_to_gen)} samples...")
    results, indices_to_gen = generate_batch(indices_to_gen)
    all_results.update(results)

if indices_to_gen:
    print(f"\n⚠️  {len(indices_to_gen)} rows still truncated after {MAX_RETRIES} retries.")
    batch_prompts = [prompts[i] for i in indices_to_gen]
    outputs = llm.generate(batch_prompts, sampling_params)
    for idx, output in zip(indices_to_gen, outputs):
        all_results[idx] = output.outputs[0].text + "  [TRUNCATED]"

# ============================================================
# PARSE THINKING vs CONTENT
# ============================================================
THINK_END_TOKEN_ID = 151668

def parse_output(text):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    try:
        index = len(token_ids) - token_ids[::-1].index(THINK_END_TOKEN_ID)
        thinking = tokenizer.decode(token_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(token_ids[index:], skip_special_tokens=True).strip("\n")
    except ValueError:
        thinking, content = "", text.strip("\n")
    return thinking, content

print("\nParsing thinking content...")
thinking_list, content_list = [], []
for i in range(len(df)):
    t, c = parse_output(all_results.get(i, ""))
    thinking_list.append(t)
    content_list.append(c)

df["raw_output"] = [all_results.get(i, "") for i in range(len(df))]
df["thinking_content"] = thinking_list
df["content"] = content_list

# ============================================================
# SAVE
# ============================================================
print(f"\nSaving to {OUTPUT_PATH}...")
df.to_parquet(OUTPUT_PATH, index=False)
print("Done! ✅")

print(f"\n--- Stats ---")
print(f"Math: {len(df_math)} rows")
print(f"Code: {len(df_code)} rows")
print(f"Total saved: {len(df)} rows")
print(f"ability distribution:\n{df['ability'].value_counts()}")
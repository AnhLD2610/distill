import pandas as pd
import pprint

df = pd.read_parquet("/home/ubuntu/Working/neurips/train.parquet")

def print_full_example(teacher_value: str):
    mask = df["extra_info"].apply(
        lambda x: isinstance(x, dict) and x.get("opd_teacher") == teacher_value
    )
    sub = df[mask]

    if sub.empty:
        print(f"❌ No example found for {teacher_value}")
        return

    row = sub.iloc[0]

    print("\n" + "="*100)
    print(f"FULL EXAMPLE FOR: {teacher_value}")
    print("="*100)

    # Convert entire row to dict
    row_dict = row.to_dict()

    # Pretty print (rõ, không bị cắt)
    pprint.pprint(row_dict, width=150)

print_full_example("math")
print_full_example("code")
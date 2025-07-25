import json
import random


def split_jsonl_equal(input_file, output_prefix):
    """
    Split a JSONL file into three equal parts.

    Args:
        input_file: Path to input JSONL file
        output_prefix: Prefix for output files
    """
    # Read all data
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # Shuffle randomly
    random.shuffle(data)

    # Calculate split size
    total = len(data)
    split_size = total // 3

    # Split data
    split1 = data[:split_size]
    split2 = data[split_size:2 * split_size]
    split3 = data[2 * split_size:]

    # Write splits
    splits = [
        (f"{output_prefix}_split-1-of-3.jsonl", split1),
        (f"{output_prefix}_split-2-of-3.jsonl", split2),
        (f"{output_prefix}_split-3-of-3.jsonl", split3)
    ]

    for filename, split_data in splits:
        with open(filename, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Split complete:")
    print(f"  Split 1: {len(split1)} samples")
    print(f"  Split 2: {len(split2)} samples")
    print(f"  Split 3: {len(split3)} samples")


# Usage
if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    random.seed(42)

    # Split the file
    split_jsonl_equal(
        input_file="open-math-reasoning-cot-3m.jsonl",
        output_prefix="open-math-reasoning-cot-2m"
    )

import os
import csv
import sys
from natsort import natsorted

parent_dir = "."

csv_headers = [
    "model", "cardnumer", "datatype", "dataset", "request_rate",
    "prompts", "input_len", "output_len", "max_concurrency",
    "Successful requests", "Benchmark duration (s)", "Total input tokens", "Total generated tokens",
    "Request throughput (req/s)", "Output token throughput (tok/s)",
    "Mean TTFT (ms)", "Median TTFT (ms)", "P90 TTFT (ms)", "P99 TTFT (ms)",
    "Mean TPOT (ms)", "Median TPOT (ms)", "P90 TPOT (ms)", "P99 TPOT (ms)",
    "Mean ITL (ms)", "Median ITL (ms)", "P90 ITL (ms)", "P99 ITL (ms)"
]


def parse_filename(filename):
    parts = filename.replace(".log", "").split("_")
    if len(parts) < 3:
        return None

    # Map the key token in the filename to the corresponding CSV field name.
    key_map = {
        "cardnumber": "cardnumer",
        "datatype": "datatype",
        "in": "input_len",
        "out": "output_len",
        "rate": "request_rate",
        "prompts": "prompts",
        "concurrency": "max_concurrency",
    }

    result = {"model": parts[2]}
    for i, token in enumerate(parts[:-1]):
        if token in key_map:
            result[key_map[token]] = parts[i + 1]
        if token == "prompts" and i + 2 < len(parts):
            # dataset is the token right after the prompts value
            result["dataset"] = parts[i + 2]

    required = {"cardnumer", "datatype", "input_len", "output_len",
                "request_rate", "prompts", "dataset", "max_concurrency"}
    if not required.issubset(result):
        return None
    return result


def parse_log_content(parent_dir):
    result_data = {}
    try:
        with open(parent_dir, "r", encoding='utf-8') as f:
            lines = f.readlines()
            if "============ Serving Benchmark Result ============\n" not in lines:
                return None
            start_idx = lines.index("============ Serving Benchmark Result ============\n") + 1
            for line in lines[start_idx:]:
                if line.startswith("=================================================="):
                    break
                key_value = line.split(":")
                if len(key_value) == 2:
                    key = key_value[0].strip()
                    value = key_value[1].strip()
                    if key in csv_headers:
                        result_data[key] = value
    except Exception as e:
        print(f"Error reading file {parent_dir}: {e}")
        return None
    return result_data

def write_to_csv(output_file, csv_headers, data_rows):
    with open(output_file, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(data_rows)

def main():
    if len(sys.argv) >= 2:
        target_dir = sys.argv[1].rstrip(os.sep)
        if not os.path.isdir(target_dir):
            print(f"Not a directory: {target_dir}")
            sys.exit(1)
        folder_name = os.path.basename(os.path.abspath(target_dir))
        output_csv = folder_name + ".csv"
    else:
        target_dir = parent_dir
        output_csv = "serving_bench.csv"

    data_rows = []

    items = os.listdir(target_dir)
    sorted_items = natsorted(items)
    for filename in sorted_items:
        if filename.endswith(".log"):
            file_path = os.path.join(target_dir, filename)

            conditions = parse_filename(filename)
            if not conditions:
                print(f"Invalid file name format: {filename}")
                continue

            results = parse_log_content(file_path)
            if not results:
                print(f"Skipping file {filename}, no valid benchmark results.")
                continue

            data_row = {**conditions, **results}
            data_rows.append(data_row)

    write_to_csv(os.path.join(target_dir, output_csv), csv_headers, data_rows)
    print(f"Results written to {os.path.join(target_dir, output_csv)}")

if __name__ == "__main__":
    main()

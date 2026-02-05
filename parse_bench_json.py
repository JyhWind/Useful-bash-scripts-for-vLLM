import json
import os
from typing import List, Dict, Optional
import csv
import pprint

def parse_json_files(output_csv: Optional[str] = "./parsed_results.csv") -> List[Dict]:
    """
    Parse all JSON files in the current working directory, extract target fields,
    and round numeric fields to 2 decimal places.
    
    Args:
        output_csv: Optional path to save CSV results (default: ./parsed_results.csv).
                    Pass None to disable CSV output.
    
    Returns:
        List of structured parsed data, each element is a dictionary containing all target fields.
    """
    # Define target fields to extract
    target_fields = [
        "num_prompts", "max_concurrency", "completed", "duration",
        "total_input_tokens", "total_output_tokens", "request_throughput",
        "output_throughput", "total_token_throughput", "mean_ttft_ms",
        "median_ttft_ms", "mean_tpot_ms", "median_tpot_ms",
        "mean_itl_ms", "median_itl_ms"
    ]
    
    # Store all parsing results
    results = []
    current_dir = os.getcwd()  # Get current working directory
    print(f"Scanning for JSON files in current directory: {current_dir}")
    
    # Iterate all files in current directory
    for filename in os.listdir(current_dir):
        # Only process JSON files
        if filename.lower().endswith(".json"):
            json_path = os.path.join(current_dir, filename)
            print(f"Processing file: {filename}")
            
            try:
                # Read JSON file
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Extract target fields and process values
                parsed_data = {"filename": filename}  # Keep filename for traceability
                for field in target_fields:
                    # Get field value (None if field doesn't exist)
                    value = data.get(field, None)
                    
                    # Round numeric values to 2 decimal places
                    if isinstance(value, (int, float)) and value is not None:
                        parsed_data[field] = round(value, 2)
                    else:
                        parsed_data[field] = value  # Preserve non-numeric values as-is
                
                results.append(parsed_data)
            
            except json.JSONDecodeError:
                print(f"Warning: {filename} is not a valid JSON file - skipped")
            except UnicodeDecodeError:
                print(f"Warning: {filename} has encoding issues (not UTF-8) - skipped")
            except Exception as e:
                print(f"Error: Failed to process {filename} - {str(e)}")
    
    # Optional: Save results to CSV
    if output_csv:
        # Define CSV headers (filename + target fields)
        headers = ["filename"] + target_fields
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to CSV: {os.path.abspath(output_csv)}")
    
    return results

# ------------------- Usage Example -------------------
if __name__ == "__main__":
    # Optional: Customize CSV output path (set to None to disable CSV)
    CUSTOM_OUTPUT_CSV = "./parsed_results.csv"  # Default output path
    
    # Execute parsing
    parsed_results = parse_json_files(output_csv=CUSTOM_OUTPUT_CSV)
    
    # Print summary
    print("\nParsing completed!")
    print(f"Successfully processed {len(parsed_results)} valid JSON files")
    
    # Print sample result (first file) if available
    if parsed_results:
        print("\nSample parsed result (first file):")
        pprint.pprint(parsed_results[0], indent=2)

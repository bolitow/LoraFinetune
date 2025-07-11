import json
import sys

def check_json_file(filepath):
    print(f"Checking JSON file: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Try to load the entire file
            data = json.load(f)
            print(f"✓ JSON is valid")
            print(f"✓ Total records: {len(data)}")
            
            # Check structure
            if isinstance(data, list) and len(data) > 0:
                print(f"\nFirst record structure:")
                print(json.dumps(data[0], indent=2))
                
                # Check if all records have 'question' and 'answer' fields
                missing_fields = []
                for i, record in enumerate(data):
                    if not isinstance(record, dict):
                        print(f"✗ Record {i} is not a dictionary")
                        continue
                    if 'question' not in record:
                        missing_fields.append((i, 'question'))
                    if 'answer' not in record:
                        missing_fields.append((i, 'answer'))
                
                if missing_fields:
                    print(f"\n✗ Missing fields in records:")
                    for idx, field in missing_fields[:10]:  # Show first 10
                        print(f"  Record {idx}: missing '{field}'")
                else:
                    print(f"✓ All records have 'question' and 'answer' fields")
            else:
                print("✗ Data is not a list or is empty")
                
    except json.JSONDecodeError as e:
        print(f"✗ JSON decode error: {e}")
        print(f"  Error at line {e.lineno}, column {e.colno}")
        
        # Try to show the problematic part
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if e.lineno <= len(lines):
                    print(f"\nProblematic line {e.lineno}:")
                    print(lines[e.lineno - 1])
        except:
            pass
            
    except Exception as e:
        print(f"✗ Error reading file: {e}")

if __name__ == "__main__":
    check_json_file("data/instruction.json")
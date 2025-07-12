import pickle
import json

# Load the pickle file
with open('tmp_outputs.pkl', 'rb') as f:
    outputs = pickle.load(f)

print("Type of outputs:", type(outputs))
print("Number of outputs:", len(outputs))

if outputs:
    # Inspect the first output
    first_output = outputs[0]
    print("\nFirst output type:", type(first_output))
    print("First output attributes:", dir(first_output))
    
    # Check outputs attribute
    if hasattr(first_output, 'outputs'):
        print("\nNumber of outputs.outputs:", len(first_output.outputs))
        if first_output.outputs:
            first_completion = first_output.outputs[0]
            print("First completion type:", type(first_completion))
            print("First completion attributes:", dir(first_completion))
            
            # Check logprobs
            if hasattr(first_completion, 'logprobs'):
                print("\nLogprobs type:", type(first_completion.logprobs))
                print("Logprobs length:", len(first_completion.logprobs) if first_completion.logprobs else "None or empty")
                
                if first_completion.logprobs:
                    print("\nFirst few logprobs entries:")
                    for i, logprob_entry in enumerate(first_completion.logprobs[:3]):
                        print(f"\nEntry {i}:")
                        print("  Type:", type(logprob_entry))
                        if logprob_entry is not None:
                            if hasattr(logprob_entry, '__dict__'):
                                print("  Dict representation:", logprob_entry.__dict__)
                            elif isinstance(logprob_entry, dict):
                                print("  Dict keys:", list(logprob_entry.keys()) if logprob_entry else "Empty dict")
                                # Show first item if exists
                                if logprob_entry:
                                    first_key = list(logprob_entry.keys())[0]
                                    print(f"  First item [{first_key}]:", logprob_entry[first_key])
                                    if hasattr(logprob_entry[first_key], '__dict__'):
                                        print(f"    Attributes:", logprob_entry[first_key].__dict__)
                        else:
                            print("  Value: None")
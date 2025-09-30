from datasets import load_dataset

print("Loading datasets...")
hotpotqa = load_dataset("hotpot_qa", "distractor")
wikimqa = load_dataset("framolfese/2WikiMultihopQA")

print("HotpotQA structure:")
print(f"Type: {type(hotpotqa)}")
print(f"Keys: {list(hotpotqa.keys())}")
print(f"Train type: {type(hotpotqa['train'])}")
print(f"First example: {hotpotqa['train'][0]}")

print("\n2WikiMultiQA structure:")
print(f"Type: {type(wikimqa)}")
print(f"Keys: {list(wikimqa.keys())}")
print(f"Train type: {type(wikimqa['train'])}")
print(f"First example: {wikimqa['train'][0]}")
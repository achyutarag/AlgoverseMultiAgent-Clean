import re
import string
from datasets import load_dataset
import evaluate
import csv

questions = []
predictions = []
references = []

def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation/articles/extra whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

def exact_match_score(prediction, ground_truth):
    return int(prediction == ground_truth)

def f1_score(prediction, ground_truth):
    pred_tokens = prediction.split()
    truth_tokens = ground_truth.split()
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)

def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 between a prediction and ground truth string.
    """
    pred_tokens = prediction.split()
    truth_tokens = ground_truth.split()
    
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate_trivia_qa():
    print("Loading TriviaQA dataset...")
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc")

    eval_dataset = dataset["validation"].select(range(100))

    predictions = []
    references = []

    print("Processing examples...")
    for example in eval_dataset:
        question = example["question"]
        gold_value = example["answer"]["value"] if example["answer"]["value"] else ""
        prediction = normalize_text(gold_value)

        if example["answer"]["normalized_aliases"]:
            reference = normalize_text(example["answer"]["normalized_aliases"][0])
        else:
            reference = ""

        questions.append(question)
        predictions.append(prediction)
        references.append(reference)

    print("Calculating metrics...")
    em_total, f1_total = 0, 0
    for pred, ref in zip(predictions, references):
        em_total += exact_match_score(pred, ref)
        f1_total += f1_score(pred, ref)

    n = len(predictions)
    print("\nEvaluation Results:")
    print(f"Exact Match: {em_total / n:.4f}")
    print(f"F1 Score:    {f1_total / n:.4f}")
    print(f"Examples:    {n}")
    
    csv_path = "evaluation_results.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow(["Question", "Prediction", "Reference", "ExactMatch", "F1Score"])

        for q, pred, ref in zip(questions, predictions, references):
            em = 1.0 if pred == ref else 0.0
            f1 = compute_f1(pred, ref)
            writer.writerow([q, pred, ref, em, f1])

    print("\nSample Predictions:")
    for i in range(3):
        print(f"\nExample {i+1}:")
        print(f"Q: {eval_dataset[i]['question']}")
        print(f"Prediction: {predictions[i]}")
        print(f"Reference:  {references[i]}")
        print(f"Correct: {predictions[i] == references[i]}")

if __name__ == "__main__":
    evaluate_trivia_qa()

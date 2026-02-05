"""
Extract CSV from existing MCP diffusion debug report JSON file.

Usage:
    python extract_csv_from_report.py results/diffusion_debug_report_20260111_231254.json
"""

import json
import csv
import sys
from pathlib import Path

def extract_csv_from_report(json_file: str):
    """Extract CSV from MCP diffusion debug report JSON."""
    json_path = Path(json_file)
    
    if not json_path.exists():
        print(f"Error: File not found: {json_file}")
        return
    
    # Load JSON report
    with open(json_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    # Create CSV file in same directory
    csv_file = json_path.parent / f"{json_path.stem}.csv"
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Question', 'Prediction', 'Ground_Truth', 'ExactMatch', 'F1Score', 'Hops'])
        
        # Data rows
        for trace in report.get('traces', []):
            question = trace.get('question', '').replace('\n', ' ').strip()
            prediction = trace.get('prediction', '').replace('\n', ' ').strip()
            ground_truth = trace.get('ground_truth', '').replace('\n', ' ').strip()
            em = trace.get('em', 0.0)
            f1 = trace.get('f1', 0.0)
            hops = trace.get('total_hops', 0)
            
            writer.writerow([question, prediction, ground_truth, f"{em:.4f}", f"{f1:.4f}", hops])
        
        # Summary row
        writer.writerow([])
        writer.writerow(['Summary', '', '', '', '', ''])
        writer.writerow(['Exact Match', '', '', f"{report['summary']['avg_em']:.4f}", '', ''])
        writer.writerow(['F1 Score', '', '', f"{report['summary']['avg_f1']:.4f}", '', ''])
        writer.writerow(['Avg Hops', '', '', f"{report['summary']['avg_hops']:.2f}", '', ''])
        writer.writerow(['Examples', '', '', f"{report['summary']['total_questions']}", '', ''])
    
    print(f"✅ CSV extracted: {csv_file}")
    print(f"   Total questions: {report['summary']['total_questions']}")
    print(f"   Avg EM: {report['summary']['avg_em']:.4f}")
    print(f"   Avg F1: {report['summary']['avg_f1']:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_csv_from_report.py <json_report_file>")
        print("Example: python extract_csv_from_report.py results/diffusion_debug_report_20260111_231254.json")
        sys.exit(1)
    
    extract_csv_from_report(sys.argv[1])


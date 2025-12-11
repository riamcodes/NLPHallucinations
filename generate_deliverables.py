import json
import csv
import os
from datetime import datetime

# Import all question categories
from questions import (
    SIMPLE_FACTUAL_QUESTIONS,
    NEGATION_ADVERSARIAL_QUESTIONS,
    IMPOSSIBLE_NONSENSE_QUESTIONS,
    COMMONSENSE_QUESTIONS,
    OBSCURE_LONG_TAIL_FACTUAL_QUESTIONS,
    TIME_SENSITIVE_QUESTIONS,
    CONTEXT_DEPENDENT_QUESTIONS,
)

def generate_deliverables():
    # Combine all questions
    ALL_QUESTIONS = (
        SIMPLE_FACTUAL_QUESTIONS + 
        NEGATION_ADVERSARIAL_QUESTIONS + 
        IMPOSSIBLE_NONSENSE_QUESTIONS +
        COMMONSENSE_QUESTIONS + 
        OBSCURE_LONG_TAIL_FACTUAL_QUESTIONS + 
        TIME_SENSITIVE_QUESTIONS + 
        CONTEXT_DEPENDENT_QUESTIONS
    )

    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)

    # Generate JSON
    with open('outputs/question_set_v1.json', 'w') as f:
        json.dump(ALL_QUESTIONS, f, indent=2)
    
    # Generate CSV
    with open('outputs/gold_labels.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['qid', 'category', 'question', 'is_hallucination', 'explanation'])
        
        for q in ALL_QUESTIONS:
            writer.writerow([
                q['qid'], 
                q['category'], 
                q['question'], 
                q['gold_label']['is_hallucination'],
                q['gold_label']['explanation']
            ])

    # Create markdown documentation
    with open('outputs/dataset_design.md', 'w', encoding='utf-8') as f:
        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Write overview
        f.write("# Hallucination Detection Dataset Design\n\n")
        f.write("## Overview\n")
        f.write(f"- Total Questions: {len(ALL_QUESTIONS)}\n")
        f.write(f"- Date Generated: {current_date}\n\n")
        
        # Count questions per category
        f.write("## Question Categories\n")
        category_counts = {}
        for q in ALL_QUESTIONS:
            category_counts[q['category']] = category_counts.get(q['category'], 0) + 1
        
        # Write category breakdown
        for category, count in category_counts.items():
            f.write(f"- {category.replace('_', ' ').title()}: {count} questions\n")
        
        # Add some additional documentation
        f.write("\n## Hallucination Detection Methodology\n")
        f.write("This dataset is designed to test generative AI models' tendency to hallucinate by:\n")
        f.write("- Providing questions across diverse categories\n")
        f.write("- Including both factual and impossible/nonsensical questions\n")
        f.write("- Marking potential hallucination risks\n")

# If running this script directly
if __name__ == "__main__":
    generate_deliverables()
    print("Deliverables generated successfully in the 'outputs' directory.")
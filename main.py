#!/usr/bin/env python3
"""
Simple Question Answering with DistilBERT and GPT-2
Minimal implementation - load models and answer questions + generate text
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleQA:
    """Simple question-answering system with text generation."""
    
    def __init__(self):
        """Initialize the QA system."""
        self.qa_pipeline = None
        self.gpt2_model = None
        self.gpt2_tokenizer = None
        
    def load_models(self):
        """Load both DistilBERT and GPT-2 models."""
        try:
            # Load DistilBERT for QA
            logger.info("Loading DistilBERT model...")
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Load GPT-2 for text generation
            logger.info("Loading GPT-2 model...")
            self.gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
            
            # Add padding token if not present
            if self.gpt2_tokenizer.pad_token is None:
                self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
            
            logger.info("Both models loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def answer_question(self, question: str, context: str):
        """Answer a question given context."""
        if self.qa_pipeline is None:
            return {"error": "Model not loaded. Call load_models() first."}
        
        try:
            result = self.qa_pipeline(question=question, context=context)
            return {
                "question": question,
                "answer": result["answer"],
                "confidence": result["score"]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def generate_text(self, prompt: str, max_length: int = 50):
        """Generate text using GPT-2."""
        if self.gpt2_model is None or self.gpt2_tokenizer is None:
            return {"error": "GPT-2 model not loaded. Call load_models() first."}
        
        try:
            # Tokenize input
            inputs = self.gpt2_tokenizer.encode(prompt, return_tensors="pt")
            
            # Generate text
            with torch.no_grad():
                outputs = self.gpt2_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.gpt2_tokenizer.eos_token_id
                )
            
            # Decode generated text
            generated_text = self.gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "prompt": prompt,
                "generated_text": generated_text,
                "length": len(generated_text.split())
            }
        except Exception as e:
            return {"error": str(e)}
    
    def process_question(self, question: str, context: str):
        """Process a question with both QA and text generation."""
        # Get answer using DistilBERT
        qa_result = self.answer_question(question, context)
        
        if 'error' in qa_result:
            return qa_result
        
        # Generate additional text using GPT-2
        generation_prompt = f"Question: {question}\nContext: {context}\nAnswer: {qa_result['answer']}\nAdditional information:"
        generation_result = self.generate_text(generation_prompt, max_length=30)
        
        return {
            "question": question,
            "answer": qa_result["answer"],
            "confidence": qa_result["confidence"],
            "generated_text": generation_result.get("generated_text", ""),
            "generation_length": generation_result.get("length", 0)
        }

def main():
    """Main function."""
    print("=" * 60)
    print("Simple Question Answering with DistilBERT and GPT-2")
    print("=" * 60)
    
    # Initialize and load models
    qa = SimpleQA()
    if not qa.load_models():
        print("Failed to load models. Exiting.")
        return
    
    # Sample questions
    questions = [
        {
            "question": "What is the capital of France?",
            "context": "France is a country in Western Europe. Its capital and largest city is Paris."
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "context": "Romeo and Juliet is a tragedy written by William Shakespeare early in his career."
        },
        {
            "question": "Who is Dr. David Lin?",
            "context": "Dr. Lin is a professor of computer science at Southern Methodist University." 
        }
    ]
    
    print(f"\nProcessing {len(questions)} questions...\n")
    
    for i, qa_data in enumerate(questions, 1):
        print(f"Question {i}: {qa_data['question']}")
        result = qa.process_question(qa_data['question'], qa_data['context'])
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.3f}")
            
            if 'generated_text' in result:
                print(f"\nGenerated Text:")
                print(f"{result['generated_text']}")
                print(f"Generated Length: {result.get('generation_length', 0)} words")
        
        print("-" * 60)
    
    print("\nDone!")

if __name__ == "__main__":
    main()

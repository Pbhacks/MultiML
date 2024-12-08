import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import json
import spacy
import nltk
import numpy as np
import pandas as pd
import re
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer

class LegalDocumentAnalyzer:
    def __init__(self, model_name='en_core_web_sm'):
        """
        Initialize the Legal Document Analyzer with NLP components
        
        Args:
            model_name (str): Spacy language model to use
        """
        # Load SpaCy model
        try:
            self.nlp = spacy.load(model_name)
        except Exception as e:
            print(f"Error loading SpaCy model: {e}")
            self.nlp = None
        
        # Define key legal clause categories
        self.legal_categories = [
            'JURISDICTION', 
            'LIABILITY', 
            'TERMINATION', 
            'CONFIDENTIALITY', 
            'COMPENSATION'
        ]
    
    def preprocess_document(self, document: str) -> List[str]:
        """
        Preprocess legal document into sentences
        
        Args:
            document (str): Full text of the legal document
        
        Returns:
            List of preprocessed sentences
        """
        if not self.nlp:
            # Fallback sentence splitting if SpaCy fails
            return re.split(r'[.!?]+', document)
        
        doc = self.nlp(document)
        sentences = [sent.text.strip() for sent in doc.sents]
        return sentences
    
    def extract_key_entities(self, document: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from legal document
        
        Args:
            document (str): Full text of the legal document
        
        Returns:
            List of identified named entities
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(document)
        entities = [
            {
                'word': ent.text,
                'entity': ent.label_
            } for ent in doc.ents
        ]
        return entities
    
    def classify_legal_clauses(self, sentences: List[str]) -> Dict[str, List[str]]:
        """
        Classify sentences into legal clause categories
        
        Args:
            sentences (List[str]): Preprocessed document sentences
        
        Returns:
            Dictionary of classified clauses
        """
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        clause_classification = {category: [] for category in self.legal_categories}
        
        for sentence, tfidf_vector in zip(sentences, tfidf_matrix):
            # Simplified classification logic
            for category in self.legal_categories:
                if category.lower() in sentence.lower():
                    clause_classification[category].append(sentence)
        
        return clause_classification
    
    def generate_summary(self, document: str, max_length: int = 150) -> str:
        """
        Generate a concise summary of the legal document
        
        Args:
            document (str): Full text of the legal document
            max_length (int): Maximum length of summary
        
        Returns:
            Condensed document summary
        """
        # Simple extractive summarization
        sentences = self.preprocess_document(document)
        
        # Use sentence length and position as basic summary criteria
        sentences.sort(key=len, reverse=True)
        summary_sentences = sentences[:3]  # Take top 3 longest sentences
        
        return ' '.join(summary_sentences)
    
    def analyze_document(self, document: str) -> Dict[str, Any]:
        """
        Comprehensive legal document analysis
        
        Args:
            document (str): Full text of the legal document
        
        Returns:
            Comprehensive analysis report
        """
        sentences = self.preprocess_document(document)
        
        analysis_report = {
            'summary': self.generate_summary(document),
            'entities': self.extract_key_entities(document),
            'clause_classification': self.classify_legal_clauses(sentences)
        }
        
        return analysis_report

class LegalDocumentAnalyzerGUI:
    def __init__(self, master):
        """
        Initialize the GUI for Legal Document Analyzer
        
        Args:
            master (tk.Tk): Main window
        """
        self.master = master
        master.title("Legal Document Analyzer")
        master.geometry("800x600")
        
        # Create analyzer instance
        self.analyzer = LegalDocumentAnalyzer()
        
        # Create GUI components
        self.create_widgets()
    
    def create_widgets(self):
        """
        Create and layout GUI widgets
        """
        # Document Input Section
        input_frame = tk.Frame(self.master, padx=10, pady=10)
        input_frame.pack(fill=tk.X)
        
        # File Selection Button
        self.file_button = tk.Button(
            input_frame, 
            text="Select Document", 
            command=self.load_document
        )
        self.file_button.pack(side=tk.LEFT, padx=5)
        
        # Document Text Input
        self.doc_input = scrolledtext.ScrolledText(
            self.master, 
            height=10, 
            width=90, 
            wrap=tk.WORD
        )
        self.doc_input.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Analyze Button
        self.analyze_button = tk.Button(
            self.master, 
            text="Analyze Document", 
            command=self.analyze_document
        )
        self.analyze_button.pack(pady=10)
        
        # Results Display
        self.results_display = scrolledtext.ScrolledText(
            self.master, 
            height=15, 
            width=90, 
            wrap=tk.WORD
        )
        self.results_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    def load_document(self):
        """
        Open file dialog to load document
        """
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[
                    ("Text Files", "*.txt"),
                    ("All Files", "*.*")
                ]
            )
            
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as file:
                    document_text = file.read()
                
                # Clear previous input and insert new text
                self.doc_input.delete(1.0, tk.END)
                self.doc_input.insert(tk.END, document_text)
        
        except Exception as e:
            messagebox.showerror("Error", f"Could not load document: {str(e)}")
    
    def analyze_document(self):
        """
        Analyze the document and display results
        """
        try:
            # Get document text
            document_text = self.doc_input.get(1.0, tk.END).strip()
            
            if not document_text:
                messagebox.showwarning("Warning", "Please enter or load a document first.")
                return
            
            # Perform analysis
            analysis_results = self.analyzer.analyze_document(document_text)
            
            # Clear previous results
            self.results_display.delete(1.0, tk.END)
            
            # Format and display results
            self.results_display.insert(tk.END, "--- Document Analysis Results ---\n\n")
            
            # Summary
            self.results_display.insert(tk.END, "Summary:\n")
            self.results_display.insert(tk.END, f"{analysis_results['summary']}\n\n")
            
            # Entities
            self.results_display.insert(tk.END, "Key Entities:\n")
            if analysis_results['entities']:
                for entity in analysis_results['entities']:
                    self.results_display.insert(tk.END, f"- {entity['word']} ({entity['entity']})\n")
            else:
                self.results_display.insert(tk.END, "No entities found.\n")
            self.results_display.insert(tk.END, "\n")
            
            # Clause Classification
            self.results_display.insert(tk.END, "Legal Clause Classification:\n")
            has_clauses = False
            for category, clauses in analysis_results['clause_classification'].items():
                if clauses:
                    has_clauses = True
                    self.results_display.insert(tk.END, f"{category}:\n")
                    for clause in clauses:
                        self.results_display.insert(tk.END, f"- {clause}\n")
                    self.results_display.insert(tk.END, "\n")
            
            if not has_clauses:
                self.results_display.insert(tk.END, "No specific legal clauses detected.\n")
        
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Could not analyze document: {str(e)}")
            # Print full traceback for debugging
            import traceback
            traceback.print_exc()

def main():
    """
    Main function to launch the GUI
    """
    root = tk.Tk()
    app = LegalDocumentAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
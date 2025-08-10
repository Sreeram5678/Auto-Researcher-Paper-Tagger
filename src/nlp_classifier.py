"""
Natural Language Processing classifier for research paper topic detection.
"""
import re
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class ResearchPaperClassifier:
    """Classifier for automatically tagging research papers with topic categories."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        )
        self.mlb = MultiLabelBinarizer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Default categories with keywords
        self.categories = self._get_default_categories()
        
        # Load custom categories if config provided
        if config_path and config_path.exists():
            self._load_custom_categories(config_path)
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
            for data in nltk_data:
                try:
                    nltk.data.find(f'tokenizers/{data}')
                except LookupError:
                    nltk.download(data, quiet=True)
        except Exception as e:
            self.logger.warning(f"Failed to download NLTK data: {e}")
    
    def _get_default_categories(self) -> Dict[str, List[str]]:
        """Get default research categories with associated keywords."""
        return {
            'Natural Language Processing': [
                'nlp', 'natural language processing', 'text mining', 'sentiment analysis',
                'machine translation', 'named entity recognition', 'part-of-speech tagging',
                'language model', 'text classification', 'information extraction',
                'question answering', 'dialogue system', 'chatbot', 'text generation',
                'word embedding', 'transformer', 'bert', 'gpt', 'attention mechanism',
                'sequence-to-sequence', 'tokenization', 'parsing', 'semantic analysis',
                'discourse analysis', 'text summarization', 'linguistic', 'corpus'
            ],
            'Computer Vision': [
                'computer vision', 'image processing', 'object detection', 'image classification',
                'face recognition', 'optical character recognition', 'ocr', 'image segmentation',
                'feature extraction', 'edge detection', 'pattern recognition', 'visual',
                'convolutional neural network', 'cnn', 'image enhancement', 'video analysis',
                'motion detection', 'tracking', 'stereo vision', 'depth estimation',
                'image registration', 'medical imaging', 'remote sensing', 'augmented reality',
                'object tracking', 'scene understanding', 'visual recognition', 'pixel'
            ],
            'Machine Learning': [
                'machine learning', 'supervised learning', 'unsupervised learning',
                'semi-supervised learning', 'reinforcement learning', 'deep learning',
                'neural network', 'artificial neural network', 'backpropagation',
                'gradient descent', 'overfitting', 'cross-validation', 'feature selection',
                'dimensionality reduction', 'clustering', 'classification', 'regression',
                'ensemble method', 'random forest', 'support vector machine', 'svm',
                'decision tree', 'naive bayes', 'k-means', 'principal component analysis',
                'pca', 'linear regression', 'logistic regression', 'model selection'
            ],
            'Deep Learning': [
                'deep learning', 'neural network', 'convolutional neural network', 'cnn',
                'recurrent neural network', 'rnn', 'long short-term memory', 'lstm',
                'gated recurrent unit', 'gru', 'autoencoder', 'generative adversarial network',
                'gan', 'transformer', 'attention mechanism', 'residual network', 'resnet',
                'batch normalization', 'dropout', 'activation function', 'backpropagation',
                'gradient descent', 'adam optimizer', 'tensorflow', 'pytorch', 'keras',
                'deep reinforcement learning', 'variational autoencoder', 'vae'
            ],
            'Reinforcement Learning': [
                'reinforcement learning', 'q-learning', 'policy gradient', 'actor-critic',
                'markov decision process', 'mdp', 'reward function', 'exploration',
                'exploitation', 'temporal difference', 'value function', 'policy',
                'monte carlo', 'sarsa', 'deep q-network', 'dqn', 'proximal policy optimization',
                'ppo', 'trust region policy optimization', 'trpo', 'multi-agent',
                'game theory', 'bandit problem', 'dynamic programming'
            ],
            'Optimization': [
                'optimization', 'linear programming', 'nonlinear programming', 'convex optimization',
                'integer programming', 'mixed integer', 'genetic algorithm', 'simulated annealing',
                'particle swarm optimization', 'evolutionary algorithm', 'gradient descent',
                'stochastic gradient descent', 'newton method', 'quasi-newton', 'bfgs',
                'constrained optimization', 'unconstrained optimization', 'global optimization',
                'local optimization', 'metaheuristic', 'ant colony optimization', 'tabu search',
                'branch and bound', 'cutting plane', 'lagrange multiplier', 'dual problem'
            ],
            'Data Mining': [
                'data mining', 'knowledge discovery', 'association rule', 'frequent pattern',
                'anomaly detection', 'outlier detection', 'clustering', 'classification',
                'prediction', 'data preprocessing', 'feature engineering', 'data cleaning',
                'data integration', 'data warehouse', 'olap', 'big data', 'data stream',
                'sequential pattern', 'time series', 'trend analysis', 'market basket analysis',
                'recommendation system', 'collaborative filtering', 'content-based filtering'
            ],
            'Graph Neural Networks': [
                'graph neural network', 'gnn', 'graph convolutional network', 'gcn',
                'graph attention network', 'gat', 'message passing', 'node classification',
                'graph classification', 'link prediction', 'graph embedding', 'network analysis',
                'social network', 'knowledge graph', 'graph theory', 'node embedding',
                'graph isomorphism', 'graph matching', 'spectral graph theory',
                'random walk', 'pagerank', 'community detection', 'network topology'
            ],
            'Meta-Learning': [
                'meta-learning', 'learning to learn', 'few-shot learning', 'one-shot learning',
                'zero-shot learning', 'transfer learning', 'domain adaptation', 'multi-task learning',
                'model-agnostic meta-learning', 'maml', 'prototypical network', 'matching network',
                'relation network', 'memory-augmented network', 'neural turing machine',
                'differentiable neural computer', 'meta-gradient', 'gradient-based meta-learning',
                'optimization-based meta-learning', 'metric-based meta-learning'
            ],
            'Federated Learning': [
                'federated learning', 'distributed learning', 'privacy-preserving', 'differential privacy',
                'secure aggregation', 'communication efficiency', 'non-iid data', 'client selection',
                'federated averaging', 'fedavg', 'personalized federated learning', 'horizontal federated learning',
                'vertical federated learning', 'cross-silo', 'cross-device', 'byzantine-robust',
                'federated optimization', 'local update', 'global model', 'edge computing'
            ]
        }
    
    def _load_custom_categories(self, config_path: Path):
        """Load custom categories from configuration file."""
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if 'categories' in config:
                    self.categories.update(config['categories'])
                    self.logger.info(f"Loaded {len(config['categories'])} custom categories")
        except Exception as e:
            self.logger.error(f"Failed to load custom categories: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for classification.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and lemmatize
        try:
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            
            # Remove stopwords and lemmatize
            processed_tokens = []
            for token in tokens:
                if token not in stop_words and len(token) > 2:
                    lemmatized = self.lemmatizer.lemmatize(token)
                    processed_tokens.append(lemmatized)
            
            return ' '.join(processed_tokens)
        except Exception as e:
            self.logger.warning(f"Failed to preprocess text: {e}")
            return text
    
    def classify_text(self, text_sections: Dict[str, str], threshold: float = 0.3) -> List[Tuple[str, float]]:
        """
        Classify text sections into research categories.
        
        Args:
            text_sections: Dictionary containing different sections of the paper
            threshold: Minimum confidence threshold for tag assignment
            
        Returns:
            List of tuples (category, confidence_score)
        """
        # Combine relevant sections
        combined_text = ""
        for section_name, content in text_sections.items():
            if section_name in ['abstract', 'keywords', 'introduction', 'conclusion']:
                combined_text += f" {content}"
        
        # Fallback to full text if sections are empty
        if not combined_text.strip():
            combined_text = text_sections.get('full_text', '')
        
        # Preprocess text
        processed_text = self.preprocess_text(combined_text)
        
        if not processed_text:
            return []
        
        # Calculate similarity scores for each category
        category_scores = []
        
        for category, keywords in self.categories.items():
            # Create keyword text
            keyword_text = ' '.join(keywords)
            
            # Calculate TF-IDF similarity
            texts = [processed_text, keyword_text]
            # Use a simple vectorizer for single document comparison
            simple_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,  # Allow all terms for single document
                max_df=1.0  # Allow all terms for single document
            )
            tfidf_matrix = simple_vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Keyword matching score
            keyword_matches = sum(1 for keyword in keywords if keyword in processed_text.lower())
            keyword_score = min(keyword_matches / len(keywords), 1.0)
            
            # Combine scores
            combined_score = 0.7 * similarity + 0.3 * keyword_score
            
            if combined_score >= threshold:
                category_scores.append((category, combined_score))
        
        # Sort by confidence score
        category_scores.sort(key=lambda x: x[1], reverse=True)
        
        return category_scores[:5]  # Return top 5 categories
    
    def get_category_keywords(self, category: str) -> List[str]:
        """Get keywords for a specific category."""
        return self.categories.get(category, [])
    
    def add_custom_category(self, category_name: str, keywords: List[str]):
        """Add a custom category with keywords."""
        self.categories[category_name] = keywords
        self.logger.info(f"Added custom category: {category_name}")
    
    def get_available_categories(self) -> List[str]:
        """Get list of available categories."""
        return list(self.categories.keys())
    
    def save_model(self, model_path: Path):
        """Save the trained model to disk."""
        model_data = {
            'categories': self.categories,
            'vectorizer': self.vectorizer,
            'mlb': self.mlb
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: Path):
        """Load a trained model from disk."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.categories = model_data['categories']
        self.vectorizer = model_data['vectorizer']
        self.mlb = model_data['mlb']
        
        self.logger.info(f"Model loaded from {model_path}")

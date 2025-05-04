"""
Evaluation pipeline for the SHL Assessment Recommendation System.
Implements metrics like Mean Recall@k and MAP@k.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import datetime

# Import our components
from src.embeddings.faiss_wrapper import FaissIndex
from src.llm.llm_recommender import LLMRecommender

class RecommenderEvaluator:
    """
    Evaluator for the SHL Assessment Recommendation System.
    """
    
    def __init__(self, 
                 recommender: LLMRecommender,
                 eval_data_path: str,
                 results_dir: str = "evaluation_results"):
        """
        Initialize the evaluator.
        
        Args:
            recommender: The recommender system to evaluate
            eval_data_path: Path to the evaluation dataset
            results_dir: Directory to save evaluation results
        """
        self.recommender = recommender
        self.eval_data_path = eval_data_path
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Load evaluation data
        self.load_eval_data()
    
    def load_eval_data(self):
        """
        Load evaluation data from the provided path.
        Expected format: CSV or JSON with job_description and relevant_assessment_ids columns.
        """
        try:
            # Try to load as CSV
            if self.eval_data_path.endswith('.csv'):
                self.eval_data = pd.read_csv(self.eval_data_path)
            # Try to load as JSON
            elif self.eval_data_path.endswith('.json'):
                with open(self.eval_data_path, 'r') as f:
                    self.eval_data = pd.DataFrame(json.load(f))
            else:
                raise ValueError(f"Unsupported file format: {self.eval_data_path}")
            
            print(f"Loaded {len(self.eval_data)} evaluation samples")
        except Exception as e:
            print(f"Error loading evaluation data: {e}")
            # Create empty DataFrame with required columns
            self.eval_data = pd.DataFrame(columns=['job_description', 'relevant_assessment_ids'])
    
    def recall_at_k(self, relevant_ids: List[str], recommended_ids: List[str], k: int = 3) -> float:
        """
        Calculate Recall@k for a single recommendation.
        
        Args:
            relevant_ids: List of relevant assessment IDs
            recommended_ids: List of recommended assessment IDs
            k: The k in Recall@k
            
        Returns:
            Recall@k score (0-1)
        """
        if not relevant_ids:
            return 0.0
        
        # Take only top k recommendations
        top_k_recommendations = recommended_ids[:k]
        
        # Count relevant items in top k
        relevant_in_top_k = len(set(relevant_ids) & set(top_k_recommendations))
        
        # Calculate recall
        recall = relevant_in_top_k / len(relevant_ids)
        
        return recall
    
    def mean_average_precision(self, relevant_ids: List[str], recommended_ids: List[str], k: int = 3) -> float:
        """
        Calculate Mean Average Precision (MAP) at k for a single recommendation.
        
        Args:
            relevant_ids: List of relevant assessment IDs
            recommended_ids: List of recommended assessment IDs
            k: The k in MAP@k
            
        Returns:
            MAP@k score (0-1)
        """
        if not relevant_ids:
            return 0.0
        
        # Take only top k recommendations
        top_k_recommendations = recommended_ids[:k]
        
        # Calculate precision at each position where a relevant item appears
        precisions = []
        relevant_count = 0
        
        for i, assessment_id in enumerate(top_k_recommendations):
            position = i + 1  # 1-based indexing
            if assessment_id in relevant_ids:
                relevant_count += 1
                precision_at_i = relevant_count / position
                precisions.append(precision_at_i)
        
        # Average precision
        if precisions:
            average_precision = sum(precisions) / len(relevant_ids)
        else:
            average_precision = 0.0
        
        return average_precision
    
    def evaluate(self, 
                 sample_size: int = None, 
                 top_k: int = 10, 
                 rerank: bool = True,
                 metrics_k: int = 3) -> Dict[str, float]:
        """
        Evaluate the recommender system on the eval dataset.
        
        Args:
            sample_size: Number of samples to evaluate (None = all)
            top_k: Number of initial candidates to retrieve from vector search
            rerank: Whether to apply LLM reranking
            metrics_k: The k for Recall@k and MAP@k
            
        Returns:
            Dictionary of evaluation metrics
        """
        if sample_size:
            eval_samples = self.eval_data.sample(min(sample_size, len(self.eval_data)))
        else:
            eval_samples = self.eval_data
        
        recalls = []
        maps = []
        all_recommendations = []
        
        print(f"Evaluating on {len(eval_samples)} samples...")
        start_time = time.time()
        
        for _, row in tqdm(eval_samples.iterrows(), total=len(eval_samples)):
            job_description = row['job_description']
            relevant_ids = row['relevant_assessment_ids']
            
            # Ensure relevant_ids is a list
            if isinstance(relevant_ids, str):
                # Try to parse as JSON if it's a string
                try:
                    relevant_ids = json.loads(relevant_ids)
                except:
                    relevant_ids = [relevant_ids]
            
            # Get recommendations
            recommendations = self.recommender.recommend(
                job_description=job_description,
                top_k=top_k,
                rerank=rerank,
                final_results=metrics_k
            )
            
            # Extract assessment IDs from recommendations
            recommended_ids = [rec.get('id', rec.get('name', '')) for rec in recommendations]
            
            # Calculate metrics
            recall = self.recall_at_k(relevant_ids, recommended_ids, k=metrics_k)
            map_score = self.mean_average_precision(relevant_ids, recommended_ids, k=metrics_k)
            
            recalls.append(recall)
            maps.append(map_score)
            
            # Store recommendation details for analysis
            all_recommendations.append({
                'job_description': job_description,
                'relevant_ids': relevant_ids,
                'recommended_ids': recommended_ids,
                'recall': recall,
                'map': map_score
            })
        
        # Calculate mean metrics
        mean_recall = np.mean(recalls)
        mean_map = np.mean(maps)
        
        # Create evaluation summary
        elapsed_time = time.time() - start_time
        eval_summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sample_size': len(eval_samples),
            'top_k': top_k,
            'rerank': rerank,
            'metrics_k': metrics_k,
            f'mean_recall@{metrics_k}': mean_recall,
            f'mean_map@{metrics_k}': mean_map,
            'processing_time_seconds': elapsed_time,
            'processing_time_per_sample': elapsed_time / len(eval_samples) if eval_samples.shape[0] > 0 else 0
        }
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(self.results_dir, f'eval_results_{timestamp}.json')
        
        with open(results_file, 'w') as f:
            json.dump({
                'summary': eval_summary,
                'detailed_results': all_recommendations
            }, f, indent=2)
        
        print(f"Evaluation results saved to {results_file}")
        print(f"Mean Recall@{metrics_k}: {mean_recall:.4f}")
        print(f"Mean MAP@{metrics_k}: {mean_map:.4f}")
        
        # Create visualization
        self.visualize_results(eval_summary, all_recommendations, timestamp)
        
        return eval_summary
    
    def visualize_results(self, 
                         summary: Dict[str, Any], 
                         detailed_results: List[Dict[str, Any]],
                         timestamp: str):
        """
        Create visualizations of evaluation results.
        
        Args:
            summary: Evaluation summary
            detailed_results: Detailed results for each evaluation sample
            timestamp: Timestamp for file naming
        """
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract metrics
        recalls = [res['recall'] for res in detailed_results]
        maps = [res['map'] for res in detailed_results]
        
        # Plot histograms
        ax1.hist(recalls, bins=10, alpha=0.7, color='blue')
        ax1.set_title(f"Recall@{summary['metrics_k']} Distribution")
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Frequency')
        ax1.axvline(summary[f'mean_recall@{summary["metrics_k"]}'], color='red', linestyle='dashed', linewidth=2)
        
        ax2.hist(maps, bins=10, alpha=0.7, color='green')
        ax2.set_title(f"MAP@{summary['metrics_k']} Distribution")
        ax2.set_xlabel('MAP')
        ax2.set_ylabel('Frequency')
        ax2.axvline(summary[f'mean_map@{summary["metrics_k"]}'], color='red', linestyle='dashed', linewidth=2)
        
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(self.results_dir, f'eval_viz_{timestamp}.png')
        plt.savefig(fig_path)
        plt.close()
        
        print(f"Visualization saved to {fig_path}")
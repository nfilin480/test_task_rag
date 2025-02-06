from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

class HybridSearcher:
    def __init__(self, alpha: float = 0.5):
        """
        Инициализация гибридного поискового движка
        
        Args:
            alpha: Вес для комбинации скоров (0-1). 
                  1.0 = только семантический поиск
                  0.0 = только лексический поиск
        """
        self.alpha = alpha
        self.documents = []
        self.bm25 = None
        self.tokenized_documents = []
        
        # Скачиваем необходимые данные для NLTK
        nltk.download('punkt')
    
    def _tokenize(self, text: str) -> List[str]:
        """Токенизация текста"""
        return word_tokenize(text.lower())
    
    def index_documents(self, documents: List[str]):
        """Индексация документов для BM25"""
        self.documents = documents
        self.tokenized_documents = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_documents)
    
    def hybrid_search(self, 
                     query: str,
                     semantic_scores: List[float],
                     top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Выполнение гибридного поиска
        
        Args:
            query: Поисковый запрос
            semantic_scores: Семантические скоры для документов
            top_k: Количество результатов для возврата
            
        Returns:
            List[Dict]: Список документов с их скорами
        """
        # Получаем BM25 скоры
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Нормализуем скоры
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
        semantic_scores = (semantic_scores - np.min(semantic_scores)) / (np.max(semantic_scores) - np.min(semantic_scores))
        
        # Комбинируем скоры
        combined_scores = self.alpha * semantic_scores + (1 - self.alpha) * bm25_scores
        
        # Получаем топ-k результатов
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': float(combined_scores[idx]),
                'semantic_score': float(semantic_scores[idx]),
                'bm25_score': float(bm25_scores[idx])
            })
            
        return results
from typing import List, Dict
import os
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseRetriever
from llm import QwenLLM
from embedder import E5Embeddings
from query_classifier import QueryClassifier
from hybrid_searcher import HybridSearcher
import numpy as np

MODEL_PATH_CLASSIFIER = "../distilbert/checkpoint-705"
MODEL_PATH_LLM = "Qwen/Qwen2.5-7B-Instruct-1M"
MODEL_PATH_EMBEDDINGS = "intfloat/multilingual-e5-large"
CHROMA_DB_PATH = "./chroma_db"



class RAGSystem:
    def __init__(self):
        self.llm = QwenLLM(model_path=MODEL_PATH_LLM)
        self.embedder = E5Embeddings(model_path=MODEL_PATH_EMBEDDINGS)
        self.classifier = QueryClassifier(model_path=MODEL_PATH_CLASSIFIER)
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=self.embedder
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.hybrid_searcher = HybridSearcher(alpha=0.7)
        
        # Индексируем документы для гибридного поиска
        if len(self.vectorstore.get()) > 0:  # Проверяем, есть ли документы
            self._index_documents()


        
    def _index_documents(self):
        """Индексация документов для гибридного поиска"""
        # Получаем все документы из векторного хранилища
        documents = self.vectorstore.get()
        if documents:  # Проверяем, что список не пустой
            self.hybrid_searcher.index_documents(documents)
        else:
            print("Warning: No documents found for indexing")
        
    
    def process_query(self, query: str) -> Dict[str, str]:
        try:
            # Классификация запроса
            query_type = self.classifier.classify_with_threshold(query)
            print(f"Класс запроса: {query_type}")

            context = []
            
            if query_type == 0:  # Если запрос требует поиска по базе знаний
                # Получаем топ-20 документов с их скорами
                top_k = 20  # Можно настроить это значение
                semantic_scores = self.vectorstore.similarity_search_with_relevance_scores(
                    query,
                    k=top_k
                )
                
                # Разделяем документы и скоры
                docs, scores = zip(*semantic_scores)
                documents = [doc.page_content for doc in docs]
                
                # Переиндексируем документы для BM25 только для топ-K результатов
                self.hybrid_searcher.index_documents(documents)
                
                # Выполняем гибридный поиск
                hybrid_results = self.hybrid_searcher.hybrid_search(
                    query=query,
                    semantic_scores=list(scores),
                    top_k=3
                )
                
                # Используем найденные документы как контекст
                context = [result['document'] for result in hybrid_results]
                
                # Выводим отладочную информацию
                print("\nTop 3 results:")
                for i, result in enumerate(hybrid_results, 1):
                    print(f"\nResult {i}:")
                    print(f"Combined score: {result['score']:.4f}")
                    print(f"Semantic score: {result['semantic_score']:.4f}")
                    print(f"BM25 score: {result['bm25_score']:.4f}")
                    print(f"Document preview: {result['document'][:200]}...")
                
            # Генерация ответа с учетом типа запроса
            response = self.llm.generate_response(query, context)
            
            return {
                "answer": response,
                "query_type": query_type
            }
        except Exception as e:
            return {
                "error": str(e)
            }

def main():
    print("Инициализация RAG системы...")
    rag_system = RAGSystem()
    print("Система готова к работе!")
    print("Введите 'выход' для завершения работы")
    
    while True:
        query = input("\nВаш вопрос: ")
        
        if query.lower() in ['выход', 'exit', 'quit']:
            break

        if query.strip() == "":
            continue
            
        try:
            result = rag_system.process_query(query)
            if "error" in result:
                print(f"\nПроизошла ошибка: {result['error']}")
            else:
                print(f"Ответ: {result['answer']}")
        except Exception as e:
            print(f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    main()
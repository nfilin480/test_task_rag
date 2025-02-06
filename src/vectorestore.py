from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from langchain.vectorstores import Chroma  # Заменяем на Chroma
from langchain.docstore.document import Document
from typing import List, Union, Dict, Any
from tqdm import tqdm
import re
import pandas as pd
import os
import nltk

from embedder import E5Embeddings

class KnowledgeDatabase:
    def __init__(self, 
                 persist_directory: str = "./chroma_db", 
                 embeddings: E5Embeddings = None,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        self.persist_directory = persist_directory
        self.embeddings = embeddings
        self.db = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Скачиваем необходимые данные для NLTK
        nltk.download('punkt')
        nltk.download('stopwords')
        
        # Создаем директорию, если её нет
        os.makedirs(self.persist_directory, exist_ok=True)

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Разбивает текст на предложения
        
        Args:
            text (str): Входной текст
            
        Returns:
            List[str]: Список предложений
        """
        # Заменяем переносы строк на пробелы для лучшего разбиения на предложения
        text = re.sub(r'\s+', ' ', text)
        
        # Разбиваем на предложения
        sentences = sent_tokenize(text)
        
        # Очищаем предложения от лишних пробелов
        sentences = [sent.strip() for sent in sentences if sent.strip()]
        
        return sentences

    def count_tokens(self, text: str) -> int:
        """
        Подсчет количества токенов в тексте
        
        Args:
            text (str): Входной текст
            
        Returns:
            int: Количество токенов
        """
        return len(word_tokenize(text))

    def split_into_chunks(self, documents: List[str]) -> List[str]:
        """
        Разделение документов на чанки методом скользящего окна,
        обрабатывая каждый документ последовательно
        
        Args:
            documents (List[str]): Список документов
            
        Returns:
            List[str]: Список чанков
        """
        chunks = []
        
        for doc in tqdm(documents):
            sentences = self.split_into_sentences(doc)
            current_pos = 0
            
            while current_pos < len(sentences):
                current_chunk = []
                current_tokens = 0
                pos = current_pos
                
                # Собираем предложения пока не достигнем максимального размера чанка
                while pos < len(sentences):
                    sentence = sentences[pos]
                    sentence_tokens = self.count_tokens(sentence)
                    
                    # Если это первое предложение в чанке или добавление не превысит лимит
                    if not current_chunk or current_tokens + sentence_tokens <= self.chunk_size:
                        current_chunk.append(sentence)
                        current_tokens += sentence_tokens
                        pos += 1
                    else:
                        break
                
                # Если собрали хотя бы одно предложение
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Сдвигаем позицию с учетом перекрытия
                    step = max(1, len(current_chunk) - int(len(current_chunk) * self.chunk_overlap / 100))
                    current_pos += step
                else:
                    # Если не удалось собрать чанк (например, очень длинное предложение)
                    current_pos += 1
        
        return chunks

    def create_vector_store(self, documents: List[str]) -> None:
        """
        Создание векторной базы данных из документов
        
        Args:
            documents (List[str]): Список документов
        """
        try:
            if not documents:
                raise ValueError("No documents to process")
                
            print(f"Processing {len(documents)} documents...")
            
            # Разделение на чанки
            chunks = self.split_into_chunks(documents)
            print(f"Created {len(chunks)} chunks")
            
            doc_objects = [Document(page_content=text) for text in chunks]
            
            # Используем Chroma
            self.db = Chroma.from_documents(
                documents=doc_objects,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Сохраняем изменения
            self.db.persist()
            
            print("Vector database created successfully!")
            
        except Exception as e:
            print(f"Error in create_vector_store: {str(e)}")
            raise

    def load_existing_db(self) -> None:
        """Загрузка существующей базы данных"""
        try:
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print("Existing database loaded successfully!")
        except Exception as e:
            raise ValueError(f"Error loading database: {str(e)}")

    def load_from_csv(self, csv_path: str, text_column: str) -> None:
        """
        Загрузка документов из CSV файла и создание векторной базы данных
        
        Args:
            csv_path (str): Путь к CSV файлу
            text_column (str): Название столбца, содержащего текстовые данные
        """
        try:
            # Загружаем CSV файл
            df = pd.read_csv(csv_path)
            
            # Проверяем наличие указанного столбца
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in CSV file")
            
            # Извлекаем тексты из указанного столбца
            documents = df[text_column].dropna().tolist()
            
            # Создаем векторную базу данных
            self.create_vector_store(documents)
            
        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
            raise

    def search_documents(self, queries: Union[str, List[str]], k: int = 5) -> List[Dict[str, Any]]:
        """
        Поиск уникальных документов по одному или нескольким запросам
        
        Args:
            queries (Union[str, List[str]]): Текст запроса или список запросов
            k (int): Количество результатов для возврата для каждого запроса
            
        Returns:
            List[Dict[str, Any]]: Список уникальных документов с метаданными
        """
        if self.db is None:
            raise ValueError("Database not initialized. Please create or load the database first.")
        
        # Преобразуем одиночный запрос в список
        if isinstance(queries, str):
            queries = [queries]
        
        # Множество для хранения уникальных документов
        unique_docs = set()
        results = []
        
        # Получаем результаты для каждого запроса
        for query in queries:
            query_results = self.db.similarity_search(query, k=k)
            for doc in query_results:
                if doc.page_content not in unique_docs:
                    unique_docs.add(doc.page_content)
                    results.append({
                        'document': doc.page_content,
                        'matched_query': query
                    })
        
        return results
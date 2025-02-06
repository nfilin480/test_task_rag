from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
import torch
from sentence_transformers import SentenceTransformer


class E5Embeddings:
    def __init__(self, model_path="intfloat/multilingual-e5-large"):
        # Проверяем доступность CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Загружаем модель на выбранное устройство
        self.model = SentenceTransformer(model_path)
        self.model.to(self.device)
        
    def preprocess_text(self, text: str) -> str:
        return "passage: " + text.lower().strip()
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            if not texts:
                raise ValueError("Empty text list provided")
                
            texts = [self.preprocess_text(str(text)) for text in texts]
            
            # Указываем batch_size для контроля потребления памяти
            embeddings = self.model.encode(
                texts, 
                convert_to_tensor=True,  # Изменено на True для использования CUDA
                device=self.device,      # Явно указываем устройство
                batch_size=32,           # Можно увеличить при использовании GPU
                show_progress_bar=True
            )
            
            # Переносим тензор на CPU и конвертируем в список
            return embeddings.cpu().numpy().tolist()
            
        except Exception as e:
            print(f"Error in embed_documents: {str(e)}")
            raise
            
    def embed_query(self, text: str) -> List[float]:
        try:
            text = "query: " + self.preprocess_text(str(text))
            embedding = self.model.encode(
                [text], 
                convert_to_tensor=True,  # Изменено на True для использования CUDA
                device=self.device,      # Явно указываем устройство
                show_progress_bar=False
            )
            
            # Переносим тензор на CPU и конвертируем в список
            return embedding.cpu().numpy()[0].tolist()
            
        except Exception as e:
            print(f"Error in embed_query: {str(e)}")
            raise
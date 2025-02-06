import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

class QueryClassifier:
    def __init__(self, model_path: str):
        """
        Инициализация классификатора запросов.
        
        Args:
            model_path (str): Путь к сохраненной модели BERT
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path) # "distilbert/distilbert-base-uncased")#
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def classify(self, text: str) -> int:
        """
        Классифицирует входной текст.
        
        Args:
            text (str): Входной текст для классификации
            
        Returns:
            int: 0 или 1 (результат классификации)
        """
        # Подготовка входных данных
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Перемещаем входные данные на нужное устройство
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Получаем предсказание
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            
        return predicted_class

    def classify_batch(self, texts: list[str]) -> list[int]:
        """
        Классифицирует batch текстов.
        
        Args:
            texts (list[str]): Список текстов для классификации
            
        Returns:
            list[int]: Список результатов классификации (0 или 1)
        """
        # Подготовка входных данных
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Перемещаем входные данные на нужное устройство
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Получаем предсказания
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_classes = torch.argmax(predictions, dim=-1).tolist()
            
        return predicted_classes
    
    def classify_with_threshold(self, text: str, threshold: float = 0.9) -> tuple[int, float]:
        """
        Классифицирует текст с учетом порога вероятности.
        
        Args:
            text (str): Входной текст для классификации
            threshold (float): Пороговое значение для положительного класса (по умолчанию 0.9)
            
        Returns:
            tuple[int, float]: Кортеж (метка класса, вероятность положительного класса)
        """
        # Подготовка входных данных
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Перемещаем входные данные на нужное устройство
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Получаем предсказание
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            positive_prob = probabilities[0][1].item()  # Вероятность положительного класса
            predicted_class = 1 if positive_prob >= threshold else 0
            
        return predicted_class
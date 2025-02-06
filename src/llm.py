from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Optional
from threading import Lock

class QwenLLM:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-7B-Instruct-1M", device: str = None):
        """
        Инициализация модели Qwen с использованием Transformers.
        
        Args:
            model_name: Название модели из HuggingFace
            device: Устройство для инференса ('cuda', 'cpu', или None для автоопределения)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Загрузка токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Загрузка модели с оптимизациями
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # Используем bfloat16 для лучшей производительности
            device_map="auto"  # Автоматическое распределение по доступным GPU
        )
        
        # Параметры генерации
        self.generation_config = {
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "repetition_penalty": 1.05,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Мьютекс для thread-safe генерации
        self.lock = Lock()
        
    def prepare_messages(self, 
                        query: str, 
                        context: Optional[List[str]] = None,
                        chat_history: Optional[List[Dict]] = None) -> str:
        """
        Подготовка сообщений для модели с учетом контекста и истории.
        """
        messages = []
        
        # Системное сообщение
        system_message = "You are Qwen - a helpful assistant."
        if context:
            system_message += "\nUse the following context to answer the question:\n"
            system_message += "\n".join(context)
        messages.append({"role": "system", "content": system_message})
        
        # Добавление истории чата
        if chat_history:
            for msg in chat_history[-3:]:  # Последние 3 обмена
                messages.append({"role": "user", "content": msg.get("input", "")})
                messages.append({"role": "assistant", "content": msg.get("output", "")})
        
        # Добавление текущего запроса
        messages.append({"role": "user", "content": query})
        
        # Применение шаблона чата
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    @torch.inference_mode()
    def generate_response(self, 
                         query: str, 
                         context: Optional[List[str]] = None,
                         chat_history: Optional[List[Dict]] = None) -> str:
        """
        Генерация ответа с использованием модели.
        
        Args:
            query: Текст запроса
            context: Список релевантных документов
            chat_history: История диалога
            
        Returns:
            str: Сгенерированный ответ
        """
        with self.lock:  # Thread-safe генерация
            try:
                # Подготовка входных данных
                prompt = self.prepare_messages(query, context, chat_history)
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=3072  # Ограничиваем длину входного контекста
                ).to(self.device)
                
                # Генерация ответа
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config,
                    use_cache=True  # Используем кэширование для ускорения
                )
                
                # Декодирование ответа
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],  # Убираем входной prompt
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                return response.strip()
                
            except Exception as e:
                print(f"Error in generate_response: {str(e)}")
                return f"An error occurred during generation: {str(e)}"
    
    def update_generation_config(self, **kwargs):
        """
        Обновление параметров генерации.
        
        Example:
            llm.update_generation_config(
                temperature=0.8,
                top_p=0.9,
                max_new_tokens=1024
            )
        """
        self.generation_config.update(kwargs)
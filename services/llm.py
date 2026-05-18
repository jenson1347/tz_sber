import logging
from gigachat import GigaChat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GigaChatCommonService")

class GigaChatIntentService:

    
    def __init__(self):
        """
        Инициализация сервиса под общедоступное API GigaChat.
        credentials: твой Authorization ключ (Base64 строка из личного кабинета)
        """
        self.credentials = 'MDE5ZTExOGYtODI3NS03NmRlLWI1OGEtYjg1YTNlNjkzYjYyOjU5ZTA1NmRiLWEyNGQtNGQzMC1iZDg4LWM5NWRkMGU1NDAzYg=='
        # Используем базовую модель, доступную всем по умолчанию

    def _generate_prompt(self, user_text: str, available_intents: list) -> str:
        """
        Генерация строгого Few-Shot промпта для классификации.
        """
        intents_list = "\n".join([f"- {intent}" for intent in available_intents])
        
        prompt = f"""Ты — интеллектуальный ассистент маршрутизации запросов в банке.
Твоя задача — проанализировать текст обращения клиента и строго соотнести его с ОДНИМ из разрешенных интентов.

### РАЗРЕШЕННЫЕ ИНТЕНТЫ:
{intents_list}

### ПРИМЕРЫ:
Запрос: "блин у меня пластик зажевало в терминале че делать"
Интент: Блокировка карты

Запрос: "подскажите где посмотреть реквизиты для перевода из другого банка"
Интент: Узнать реквизиты

Запрос: "хочу закрыть вклад и забрать наличные"
Интент: Закрытие счета

### ТЕКУЩИЙ ЗАПРОС КЛИЕНТА:
Запрос: "{user_text}"

Выведи ТОЛЬКО название интента из списка выше. Никаких лишних слов, приветствий и объяснений.
Интент:"""
        return prompt

    def get_intent(self, text: str, available_intents: list) -> str:
        """
        Метод отправки запроса в общее API GigaChat через безопасное подключение.
        """
        logger.info(f"Отправка запроса в GigaChat для текста: '{text[:30]}...'")
        prompt = self._generate_prompt(text, available_intents)
        
        try:
            # Открываем безопасную сессию к общему API Сбера
            with GigaChat(credentials=self.credentials, verify_ssl_certs=False) as giga:
                response = giga.chat({
                    "model": "GigaChat", # Строго базовая модель "GigaChat"
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0, # Убираем фантазию модели, нам нужна точность
                    "max_tokens": 40
                })
                
            raw_answer = response.choices[0].message.content.strip()
            
            # Валидация ответа
            if raw_answer in available_intents:
                return raw_answer
                
            # Проверка на частичное совпадение (если модель добавила точку или кавычки)
            for intent in available_intents:
                if intent.lower() in raw_answer.lower():
                    return intent

            print(raw_answer)
                    
            return raw_answer
            
        except Exception as e:
            logger.error(f"Сбой общего API GigaChat: {str(e)}")
            raise e

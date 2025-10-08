# main.py
import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import openai
import requests
import uvicorn

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Модели данных Pydantic
class PostRequest(BaseModel):
    topic: str = Field(..., description="Тема для генерации поста", example="искусственный интеллект")
    include_news: bool = Field(True, description="Включать ли актуальные новости в контекст")
    max_tokens: int = Field(500, description="Максимальное количество токенов для генерации", ge=50, le=2000)

class PostResponse(BaseModel):
    title: str
    meta_description: str
    post_content: str
    news_context: Optional[List[str]] = None

class HealthCheck(BaseModel):
    status: str
    openai_status: str
    currents_status: str

# Инициализация FastAPI приложения
app = FastAPI(
    title="Blog Post Generator API",
    description="API для генерации блог-постов с использованием OpenAI и актуальных новостей",
    version="1.0.0"
)

# Конфигурация
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CURRENTS_API_KEY = os.getenv("CURRENTS_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    CURRENTS_API_URL = "https://api.currentsapi.services/v1/latest-news"

# Проверка наличия API ключей
if not Config.OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY не найден в переменных окружения")
if not Config.CURRENTS_API_KEY:
    logger.warning("CURRENTS_API_KEY не найден - функциональность новостей будет отключена")

# Инициализация клиента OpenAI
client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)

class CurrentsAPI:
    """Класс для работы с Currents API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = Config.CURRENTS_API_URL
    
    def get_news_by_topic(self, topic: str, limit: int = 5) -> List[str]:
        """
        Получение актуальных новостей по теме
        """
        if not self.api_key:
            logger.warning("Currents API ключ не настроен")
            return []
        
        try:
            params = {
                "keywords": topic,
                "language": "ru",
                "limit": limit,
                "apiKey": self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            news_items = []
            
            for article in data.get("news", [])[:limit]:
                title = article.get("title", "")
                description = article.get("description", "")
                if title:
                    news_items.append(f"{title}: {description}")
            
            logger.info(f"Получено {len(news_items)} новостей по теме '{topic}'")
            return news_items
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при получении новостей: {e}")
            return []
        except Exception as e:
            logger.error(f"Неожиданная ошибка в Currents API: {e}")
            return []

class PostGenerator:
    """Класс для генерации постов"""
    
    def __init__(self):
        self.openai_client = client
        self.news_client = CurrentsAPI(Config.CURRENTS_API_KEY)
    
    async def generate_post(self, request: PostRequest) -> PostResponse:
        """
        Генерация полного поста с заголовком, мета-описанием и контентом
        """
        try:
            # Получение актуальных новостей по теме
            news_context = []
            if request.include_news and Config.CURRENTS_API_KEY:
                news_context = self.news_client.get_news_by_topic(request.topic)
            
            # Генерация заголовка
            title = await self._generate_title(request.topic, news_context)
            
            # Генерация мета-описания
            meta_description = await self._generate_meta_description(title, news_context)
            
            # Генерация содержания поста
            post_content = await self._generate_post_content(
                request.topic, title, news_context, request.max_tokens
            )
            
            return PostResponse(
                title=title,
                meta_description=meta_description,
                post_content=post_content,
                news_context=news_context if news_context else None
            )
            
        except openai.APIError as e:
            logger.error(f"OpenAI API ошибка: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Ошибка сервиса генерации контента: {e}"
            )
        except Exception as e:
            logger.error(f"Неожиданная ошибка при генерации поста: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Внутренняя ошибка сервера"
            )
    
    async def _generate_title(self, topic: str, news_context: List[str]) -> str:
        """Генерация привлекательного заголовка"""
        context_prompt = self._build_news_context(news_context)
        
        prompt = f"""
        Придумай привлекательный и SEO-оптимизированный заголовок для поста на тему: "{topic}"
        {context_prompt}
        
        Требования:
        - Максимально привлекательный и кликабельный
        - Включает ключевые слова по теме
        - Не более 10 слов
        """
        
        response = self.openai_client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.8,
        )
        
        return response.choices[0].message.content.strip()
    
    async def _generate_meta_description(self, title: str, news_context: List[str]) -> str:
        """Генерация мета-описания"""
        context_prompt = self._build_news_context(news_context)
        
        prompt = f"""
        Напиши краткое, но информативное мета-описание для поста с заголовком: "{title}"
        {context_prompt}
        
        Требования:
        - Не более 160 символов
        - Включает ключевые слова
        - Призывает к прочтению
        - Описывает основную пользу статьи
        """
        
        response = self.openai_client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7,
        )
        
        return response.choices[0].message.content.strip()
    
    async def _generate_post_content(self, topic: str, title: str, 
                                   news_context: List[str], max_tokens: int) -> str:
        """Генерация основного содержания поста"""
        context_prompt = self._build_news_context(news_context)
        
        prompt = f"""
        Напиши подробный и увлекательный пост для блога на тему: "{topic}"
        Заголовок поста: "{title}"
        {context_prompt}
        
        Требования к структуре:
        - Введение с hook-зацепкой
        - Основная часть с подзаголовками
        - Практические примеры и кейсы
        - Заключение с выводами
        - Используй короткие абзацы
        - Добавь ключевые слова для SEO
        - Будь информативным и полезным
        
        Учитывай актуальные новости и тренды в контенте.
        """
        
        response = self.openai_client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        
        return response.choices[0].message.content.strip()
    
    def _build_news_context(self, news_context: List[str]) -> str:
        """Построение контекста из новостей для промпта"""
        if not news_context:
            return ""
        
        context = "\nАктуальные новости по теме:\n"
        for i, news in enumerate(news_context, 1):
            context += f"{i}. {news}\n"
        
        context += "\nУчти эти новости при создании контента, сделай отсылки где это уместно."
        return context

# Инициализация генератора
post_generator = PostGenerator()

# Эндпоинты FastAPI
@app.get("/", response_model=dict)
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Blog Post Generator API", 
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Проверка работоспособности сервиса и API"""
    try:
        # Проверка OpenAI
        openai_status = "healthy"
        try:
            client.models.list()
        except Exception as e:
            openai_status = f"unhealthy: {e}"
            logger.error(f"OpenAI health check failed: {e}")
        
        # Проверка Currents API
        currents_status = "healthy" if Config.CURRENTS_API_KEY else "not_configured"
        if Config.CURRENTS_API_KEY:
            try:
                test_news = CurrentsAPI(Config.CURRENTS_API_KEY).get_news_by_topic("technology", 1)
                if not test_news:
                    currents_status = "no_connection"
            except Exception as e:
                currents_status = f"unhealthy: {e}"
                logger.error(f"Currents API health check failed: {e}")
        
        return HealthCheck(
            status="service_running",
            openai_status=openai_status,
            currents_status=currents_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

@app.post("/generate-post", response_model=PostResponse)
async def generate_post_endpoint(request: PostRequest):
    """
    Генерация полного блог-поста по заданной теме
    """
    logger.info(f"Генерация поста для темы: {request.topic}")
    return await post_generator.generate_post(request)

@app.get("/topics/suggestions", response_model=List[str])
async def get_topic_suggestions():
    """Получение списка популярных тем для постов"""
    suggestions = [
        "Преимущества медитации для ментального здоровья",
        "Здоровое питание для занятых людей",
        "Советы по управлению временем и продуктивности",
        "Как начать свой бизнес с нуля",
        "Путешествия по бюджету: лайфхаки",
        "Искусственный интеллект в повседневной жизни",
        "Цифровой детокс: зачем и как",
        "Устойчивое развитие и экологичный образ жизни",
        "Удаленная работа: плюсы и минусы",
        "Инвестиции для начинающих"
    ]
    return suggestions

@app.get("/config")
async def get_config():
    """Получение текущей конфигурации (без секретных ключей)"""
    return {
        "openai_model": Config.OPENAI_MODEL,
        "currents_api_configured": bool(Config.CURRENTS_API_KEY),
        "max_tokens_limit": 2000
    }

# Запуск приложения
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Только для разработки
        log_level="info"
    )

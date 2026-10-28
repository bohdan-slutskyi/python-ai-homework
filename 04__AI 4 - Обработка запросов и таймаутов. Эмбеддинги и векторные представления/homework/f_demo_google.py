"""Семантический поиск с актуальным Gemini SDK и устойчивыми API-запросами."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_random_exponential


EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSION = 768
TOP_K = 4
REQUEST_TIMEOUT_MS = 20_000
# Ограничиваем частоту собственных запросов до 60 в минуту.
MIN_REQUEST_INTERVAL_SECONDS = 1.0

TEXTS_TO_INDEX = [
    "Кошка сидит на окне.",
    "Собака играет в парке.",
    "Ананас растёт в тропиках.",
    "Кот спит на диване.",
    "Пёс лает на почтальона.",
    "Фрукт ананас очень вкусный.",
    "Домашняя кошка любит ласку.",
    "Верный пёс охраняет дом.",
    "Спелый ананас полон витаминов.",
    "Кошки любят рыбу и мясо.",
    "Основной рацион кошек — белок.",
    "Чем кормить котёнка?",
    "Лучший корм для кошек — сбалансированный.",
    "Коты едят сухой и влажный корм.",
    "Нельзя кормить кошку шоколадом.",
    "Молоко не всегда полезно для кошек.",
    "Я обожаю программировать.",
    "Программирование — это то, что мне очень нравится.",
    "Меня увлекает разработка программного обеспечения.",
    "Я испытываю интерес к написанию кода.",
    "Программирование приносит мне удовольствие.",
    "Мне интересно заниматься программированием.",
    "Моё хобби — кодинг.",
    "В свободное время я занимаюсь программированием.",
    "Кодинг — моё любимое увлечение.",
    "Я увлекаюсь кодингом на досуге.",
    "Программирование — моё хобби и страсть.",
    "Когда есть свободное время, я кодирую.",
    "Кодинг — хобби, которым я наслаждаюсь.",
    "Птицы строят гнёзда на деревьях.",
    "Зимой многие птицы улетают на юг.",
    "Велосипед удобен для коротких поездок.",
    "Автобус следует по городскому маршруту.",
    "Поезд быстро доставляет пассажиров в другой город.",
    "Свежие овощи нужны для здорового питания.",
    "Для приготовления супа нарезают овощи.",
    "Осенью листья меняют цвет и опадают.",
    "Солнечная погода подходит для прогулки.",
    "Музыка помогает сосредоточиться во время работы.",
    "Книги помогают узнавать новое.",
]

T = TypeVar("T")


class RequestRateLimiter:
    """Соблюдает минимальную паузу между вызовами внешнего API."""

    def __init__(self, minimum_interval: float) -> None:
        self.minimum_interval = minimum_interval
        self._last_request_at = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        remaining = self.minimum_interval - (now - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()


def is_retryable(error: BaseException) -> bool:
    """Повторяет только временные сетевые ошибки, таймауты, 429 и 5xx."""
    if isinstance(error, errors.APIError):
        return error.code == 429 or error.code >= 500
    return error.__class__.__name__ in {
        "TimeoutException",
        "ConnectError",
        "ReadError",
        "RemoteProtocolError",
    }


def with_retry(operation: Callable[..., T], *args: object) -> T:
    retryer = Retrying(
        retry=retry_if_exception(is_retryable),
        wait=wait_random_exponential(multiplier=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    return retryer(operation, *args)


def embed_texts(
    client: genai.Client,
    limiter: RequestRateLimiter,
    texts: list[str],
    *,
    query: bool = False,
) -> np.ndarray:
    """Получает по отдельному вектору на текст одним пакетным вызовом."""
    contents = [
        types.Content(
            parts=[
                types.Part(
                    text=(
                        f"task: search result | query: {text}"
                        if query
                        else f"title: none | text: {text}"
                    )
                )
            ]
        )
        for text in texts
    ]

    def request() -> object:
        limiter.wait()
        return client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=contents,
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSION,
            ),
        )

    response = with_retry(request)
    embeddings = getattr(response, "embeddings", None) or []
    vectors = [item.values for item in embeddings]
    if len(vectors) != len(texts):
        raise RuntimeError(
            "API не вернул отдельный эмбеддинг для каждого текста; "
            "индекс нельзя построить корректно."
        )
    return np.asarray(vectors, dtype=np.float32)


def build_index(
    client: genai.Client, limiter: RequestRateLimiter, texts: list[str]
) -> np.ndarray:
    vectors = embed_texts(client, limiter, texts)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def read_query() -> str:
    """Запрашивает у пользователя произвольную фразу для поиска."""
    while True:
        query = input("Введите произвольную строку для поиска: ").strip()
        if query:
            return query
        print("Запрос не должен быть пустым.")


def semantic_search(
    client: genai.Client,
    limiter: RequestRateLimiter,
    index: np.ndarray,
    texts: list[str],
    query: str,
    top_k: int = TOP_K,
) -> tuple[list[tuple[str, float]], str]:
    query_vector = embed_texts(client, limiter, [query], query=True)
    query_vector /= np.maximum(np.linalg.norm(query_vector, axis=1, keepdims=True), 1e-12)
    scores = index @ query_vector[0]
    positions = np.argsort(scores)[::-1][: min(top_k, len(texts))]
    results = [(texts[int(i)], float(scores[i])) for i in positions]
    combined_text = "\n".join(text for text, _ in results)
    return results, combined_text


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    key_file = (
        project_root
        / "03__AI 3 - Токенизация и работа с API LLM"
        / "homework"
        / "gemini_api"
        / ".env"
    )
    load_dotenv(key_file)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print(
            "Не найден GEMINI_API_KEY. Проверьте локальный .env из проекта AI 3.",
            file=sys.stderr,
        )
        return 1

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )
    limiter = RequestRateLimiter(MIN_REQUEST_INTERVAL_SECONDS)

    try:
        print(f"Индексация {len(TEXTS_TO_INDEX)} предложений через {EMBEDDING_MODEL}…")
        index = build_index(client, limiter, TEXTS_TO_INDEX)
        query = read_query()
        results, combined_text = semantic_search(
            client, limiter, index, TEXTS_TO_INDEX, query
        )
    except Exception as error:
        status = getattr(error, "code", None)
        print(f"Тип ошибки: {error.__class__.__name__}", file=sys.stderr)
        if status:
            print(
                f"Запрос не выполнен (HTTP {status}). Проверьте квоту и доступ API.",
                file=sys.stderr,
            )
        else:
            print(
                "Запрос не выполнен. Проверьте подключение и доступность API.",
                file=sys.stderr,
            )
        return 1

    print(f"\nТоп-{len(results)} совпадений:")
    for text, score in results:
        print(f"- {text} (сходство: {score:.3f})")
    print("\nОбъединённый текст top-k:")
    print(combined_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

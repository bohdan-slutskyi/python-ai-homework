"""Один текстовый запрос к Gemini API с ключом из .env или окружения."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re
import sys

from dotenv import load_dotenv
import requests


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT = (
    "Объясни простыми словами, что такое токен в языковой модели. "
    "Ответь по-русски в 2–3 предложениях."
)


def generate_content(api_key: str, model: str, prompt: str) -> dict:
    if not api_key.strip():
        raise ValueError("Не задан GEMINI_API_KEY в .env или переменных окружения.")
    if not re.fullmatch(r"gemini-[A-Za-z0-9.-]+", model):
        raise ValueError("Некорректный идентификатор GEMINI_MODEL.")
    if not prompt.strip():
        raise ValueError("Текст запроса не должен быть пустым.")

    # Ключ передаётся в заголовке: он не попадает в URL и вывод программы.
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": api_key},
        json={
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 1024},
        },
        timeout=30,
    )
    if response.status_code != 200:
        # Сырое тело ошибки может содержать служебные сведения, поэтому его не печатаем.
        raise RuntimeError(f"Gemini API вернул HTTP {response.status_code}.")
    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini не вернул вариантов ответа; проверьте запрос и ограничения API.")
    candidate = candidates[0]
    if candidate.get("finishReason") != "STOP":
        raise RuntimeError("Генерация не завершена обычным образом; успешный результат не сохранён.")
    text = "\n".join(
        part["text"]
        for part in candidate.get("content", {}).get("parts", [])
        if "text" in part and not part.get("thought", False)
    ).strip()
    if not text:
        raise RuntimeError("Gemini вернул пустой текст; успешный результат не сохранён.")
    return {
        "model": model,
        "prompt": prompt,
        "response": text,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "usage": data.get("usageMetadata", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()
    load_dotenv(BASE_DIR / ".env", override=False)
    try:
        result = generate_content(
            os.getenv("GEMINI_API_KEY", ""),
            os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            args.prompt,
        )
    except (ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except requests.RequestException:
        print("Сетевой запрос к Gemini не выполнен. Проверьте соединение и повторите запуск.", file=sys.stderr)
        return 1

    output_dir = BASE_DIR / "results"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "response.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Модель: {result['model']}")
    print(f"Запрос: {result['prompt']}")
    print("\nОтвет Gemini:")
    print(result["response"])
    print("\nУспешный ответ сохранён: results/response.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Клиент ЛОКАЛЬНОЙ модели через OpenAI-совместимый API.

Провайдер ``local`` обращается к локальному серверу (llama.cpp
``llama-server``, Ollama, LM Studio), который отдаёт стандартный
``POST /v1/chat/completions``. Формат ответа совпадает с облачными
провайдерами проекта (DeepSeek/GigaChat), поэтому агент использует его
единообразно.

Отличия от облачных провайдеров:
  * URL задаётся пользователем (http://localhost:8080/v1);
  * ключ авторизации ОБЫЧНО не нужен (llama.cpp локально) — если задан,
    добавляется как ``Authorization: Bearer <key>``;
  * ``usage`` может отсутствовать — тогда токены оцениваются по длине
    текста (грубо, для статистики).
"""
import json
import urllib.error
import urllib.request

from . import config

__all__ = ["chat", "list_models"]


def _chat_url(base_url):
    """Возвращает полный URL chat/completions из базового (…/v1)."""
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        base = "http://%s:%d/v1" % (config.LOCAL_HOST, config.LOCAL_PORT)
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _models_url(base_url):
    """URL списка моделей (…/v1/models)."""
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        base = "http://%s:%d/v1" % (config.LOCAL_HOST, config.LOCAL_PORT)
    return base + "/models"


def _estimate_tokens(text):
    """Грубая оценка токенов по длине (русский ~2–3 символа/токен)."""
    return max(1, len(str(text or "")) // 3)


def chat(messages, model=None, base_url=None, api_key=None, temperature=None,
         max_tokens=None, timeout=None):
    """Отправляет сообщения локальному серверу и возвращает ответ.

    Возвращает dict: content, prompt_tokens, completion_tokens — как у
    DeepSeek/GigaChat. Исключения пробрасываются наружу (обрабатываются
    вызывающим кодом так же, как у других провайдеров).
    """
    url = _chat_url(base_url)
    payload = {"model": model or config.LOCAL_LABEL,
               "messages": messages, "stream": False}
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    headers = {"Content-Type": "application/json"}
    key = str(api_key or "").strip()
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers, method="POST")
    with urllib.request.urlopen(
            req, timeout=timeout or config.LOCAL_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    p_tok = int(usage.get("prompt_tokens", 0) or 0)
    c_tok = int(usage.get("completion_tokens", 0) or 0)
    if not p_tok and not c_tok:
        # Локальный сервер мог не вернуть usage — оцениваем по длине.
        p_tok = _estimate_tokens(" ".join(str(m.get("content", ""))
                                          for m in messages))
        c_tok = _estimate_tokens(content)
    return {"content": content, "prompt_tokens": p_tok,
            "completion_tokens": c_tok}


def list_models(base_url=None, timeout=5):
    """Список моделей локального сервера (GET /v1/models).

    Возвращает (models, error): models — список id, error — строка или None.
    """
    try:
        req = urllib.request.Request(_models_url(base_url), method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = data.get("data") or []
        models = [str(it.get("id")) for it in items
                  if isinstance(it, dict) and it.get("id")]
        return models, None
    except urllib.error.HTTPError as exc:
        return [], "HTTP %s" % exc.code
    except Exception as exc:                          # noqa: BLE001
        return [], str(exc)

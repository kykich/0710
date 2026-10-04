# -*- coding: utf-8 -*-
"""LLM-части улучшенного RAG: query rewrite и LLM-реранкер.

Здесь живут две функции, которым нужен ДОСТУП К LLM (в отличие от
rtk_app/rag.py, который остаётся «чистым»: эмбеддинги + лексика + порог).

  * rewrite_query(agent, question, ...) — переформулирует запрос в несколько
    поисковых формулировок (синонимы/профессиональные термины). Возвращает
    НАБОР строк для поиска по индексу.
  * rag_rerank(agent, query, hits, ...) — пересортировывает найденные
    фрагменты по релевантности силами LLM.

Модуль не хранит состояние: agent нужен только как «прокси» к LLM
(agent._resolve_model / agent._chat_text / agent._chat_json).
"""
from . import config

__all__ = ["rewrite_query", "rag_rerank"]


def _clean_line(line):
    """Убирает маркеры списка/нумерацию из строки ответа модели."""
    s = str(line or "").strip()
    # Срезаем префиксы вида "- ", "* ", "1. ", "1) ".
    while s and s[0] in "-*0123456789.)":
        s = s[1:].strip()
    return s


def rewrite_query(agent, question, model=None, count=None):
    """РАСШИРЯЕТ поисковый запрос силами LLM (query rewrite).

    Возвращает СПИСОК формулировок (включая полезные перефразировки), без
    исходного запроса (его добавит вызывающий). Пустой список — при сбое или
    когда перефразировать нечего.
    """
    question = str(question or "").strip()
    if not question:
        return []
    n = int(count or config.RAG_REWRITE_COUNT)
    provider, model_name = agent._resolve_model(model or config.RAG_REWRITE_MODEL)
    rules = (
        "Ты — модуль расширения поискового запроса для базы знаний (RAG) "
        "по инженерной/технической документации. Переформулируй запрос в "
        + str(n) + " КОРОТКИХ поисковых формулировок, которые помогут найти "
        "нужный раздел документа. Правила:\n"
        "1) используй синонимы и ПРОФЕССИОНАЛЬНЫЕ термины (например, "
        "«обратная засечка» -> «resection», «координатная засечка», "
        "«станция», «геодезические измерения»);\n"
        "2) добавь возможные названия процедур/функций прибора;\n"
        "3) каждая формулировка — отдельная строка, без нумерации и "
        "пояснений, только сам поисковый запрос;\n"
        "4) язык — русский (термины можно на английском).\n\n"
        "Запрос: " + question
    )
    try:
        res = agent._chat_text(provider, model_name,
                               [{"role": "user", "content": rules}],
                               temperature=0.2)
        text = res.get("content", "") if isinstance(res, dict) else str(res)
    except Exception:                            # noqa: BLE001
        return []
    out = []
    for ln in str(text or "").splitlines():
        clean = _clean_line(ln)
        if clean and clean not in out:
            out.append(clean)
    return out[:n]


def rag_rerank(agent, query, hits, model=None):
    """LLM-РЕРАНКЕР: пересортировывает кандидатов RAG по релевантности.

    Возвращает СПИСОК тех же hit, отсортированный по убыванию оценки LLM;
    в каждый оценённый hit добавляется поле "rerank" (0–10). При любой
    ошибке возвращает исходный порядок.
    """
    hits = list(hits or [])
    if not hits:
        return []
    limit = int(config.RAG_RERANK_CANDIDATES)
    cand = hits[:limit]
    provider, model_name = agent._resolve_model(model or config.RAG_RERANK_MODEL)
    blocks = []
    for i, h in enumerate(cand):
        body = str(h.get("text", "")).strip().replace("\n", " ")
        if len(body) > 400:
            body = body[:400] + "…"
        full = str(h.get("file", "")).replace("\\", "/")
        src = "%s, стр. %s" % (full.split("/")[-1], h.get("page"))
        blocks.append("[%d] (%s) %s" % (i + 1, src, body))
    prompt = (
        "Оцени релевантность каждого фрагмента документа запросу "
        "пользователя по шкале от 0 до 10 (10 — фрагмент прямо отвечает "
        "на запрос, 0 — совсем не по теме).\n"
        "Запрос: " + str(query or "") + "\n\n"
        "Фрагменты:\n" + "\n".join(blocks) + "\n\n"
        'Верни СТРОГО JSON вида {"scores": [{"id": 1, "score": 8}, ...]}, '
        "где id — номер фрагмента из списка выше. Без пояснений."
    )
    try:
        data, _raw = agent._chat_json(provider, model_name, prompt,
                                      temperature=0.0, max_tokens=800)
    except Exception:                            # noqa: BLE001
        return hits
    raw_scores = data.get("scores") if isinstance(data, dict) else None
    score_map = {}
    for item in (raw_scores or []):
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("id"))
            sc = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        score_map[idx] = sc
    if not score_map:
        return hits
    for i, h in enumerate(cand):
        sc = score_map.get(i + 1)
        if sc is not None:
            h["rerank"] = sc
    cand.sort(key=lambda h: h.get("rerank", -1.0), reverse=True)
    return cand + hits[limit:]

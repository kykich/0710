# -*- coding: utf-8 -*-
"""Проверка УЛУЧШЕННОГО поиска RAG (2-й этап) — офлайн, без Ollama.

Покрывает:
  1. Токенизацию и лексический скор (tokenize_ru / lexical_score);
  2. Гибридный скор (hybrid_score): веса, клампы;
  3. Гибридное ранжирование: специфичный чанк («обратная засечка») встаёт
     ВЫШЕ «общего описания» за счёт лексического сигнала (кейс стр. 54);
  4. Фильтрацию по порогу min_score;
  5. search_multi (query rewrite): объединение по формулировкам;
  6. Реранкер: rerank() с callback и без него;
  7. Настройки 2-го этапа в RagStore (load/update/клампы);
  8. rag_llm.rewrite_query / rag_rerank с ФЕЙКОВЫМ агентом (без сети):
     * rewrite_query собирает формулировки из ответа LLM;
     * rag_rerank пересортировывает по оценкам и пишет поле "rerank";
     * при ошибке LLM — graceful degradation.

Эмбеддинги подменяются детерминированной фейковой функцией (monkeypatch),
поэтому Ollama НЕ требуется. Запуск: python -m tests.check_rag_rank
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.harness import check, section, finish, ensure_utf8  # noqa: E402
from rtk_app import rag, rag_llm  # noqa: E402


# ----------------------------------------------------------------------
# Фейковые векторы: строим «эмбеддинг» по сумме кодов символов так, чтобы
# похожие по буквам тексты были ближе. Для тестов достаточно, чтобы
# «обратная засечка» и запрос давали больший косинус к нужному чанку.
# ----------------------------------------------------------------------
def _fake_embed(text):
    text = str(text or "").lower().replace("ё", "е")
    dim = rag.config.RAG_EMBED_DIM
    v = [0.0] * dim
    for ch in text:
        v[ord(ch) % dim] += 1.0
    # Нормализуем (косинус посчитается и так, но так честнее).
    return v


class _PatchEmbed(object):
    """Контекстный менеджер: подменяет embed_text фейковой функцией."""

    def __enter__(self):
        self._orig = rag.embed_text
        rag.embed_text = _fake_embed
        return self

    def __exit__(self, *exc):
        rag.embed_text = self._orig
        return False


def _make_store_with(chunks):
    """Создаёт RagStore с готовым набором чанков (без PDF/эмбеддингов)."""
    d = tempfile.mkdtemp(prefix="rag_rank_")
    idx = os.path.join(d, "idx.db")
    setf = os.path.join(d, "set.json")
    store = rag.RagStore(index_file=idx, settings_file=setf)
    with _PatchEmbed():
        conn = store._conn()
        for i, (text, meta) in enumerate(chunks):
            vec = _fake_embed(text)
            conn.execute(
                "INSERT INTO chunks (id, file, page, chunk, text, vec) "
                "VALUES (?,?,?,?,?,?)",
                (i + 1, meta.get("file", "doc.pdf"), meta.get("page", 1),
                 i, text, rag._pack_vector(vec)))
        conn.commit()
    return store


def test_tokenize():
    section("1. Токенизация и лексический скор")
    toks = rag.tokenize_ru("Обратная засечка координаты станции")
    check("tokenize_ru вернул токены", len(toks) >= 3)
    check("стоп-слова отброшены (нет «и»/«в»)",
          "и" not in toks and "в" not in toks)
    check("регистр/ё нормализованы (ё→е)",
          rag.tokenize_ru("Ёжик") == rag.tokenize_ru("ежик"))
    check("короткие токены отброшены",
          "по" not in rag.tokenize_ru("по дороге"))
    check("пустой текст → пустой список", rag.tokenize_ru("") == [])
    check("None → пустой список", rag.tokenize_ru(None) == [])

    # Лексическое совпадение: все термины найдены → 1.0.
    q = rag.tokenize_ru("обратная засечка")
    full = "ПРОЦЕДУРА Обратная засечка выполняется для координат."
    none = "Общее описание прибора и его назначение."
    check("lexical_score: все термины есть → 1.0",
          abs(rag.lexical_score(q, full) - 1.0) < 1e-9)
    check("lexical_score: терминов нет → 0.0",
          rag.lexical_score(q, none) == 0.0)
    check("lexical_score: пустой запрос → 0.0",
          rag.lexical_score([], full) == 0.0)


def test_hybrid_score():
    section("2. Гибридный скор")
    check("hybrid: alpha=0.6, cos=0.5, lex=1 → 0.7",
          abs(rag.hybrid_score(0.5, 1.0, 0.6) - 0.7) < 1e-9)
    check("hybrid: alpha=1 → только косинус",
          abs(rag.hybrid_score(0.4, 1.0, 1.0) - 0.4) < 1e-9)
    check("hybrid: alpha=0 → только лексика",
          abs(rag.hybrid_score(0.9, 0.3, 0.0) - 0.3) < 1e-9)
    check("hybrid: отрицательный косинус клампится в 0",
          abs(rag.hybrid_score(-0.5, 0.0, 0.6) - 0.0) < 1e-9)
    check("hybrid: значения >1 клампятся",
          abs(rag.hybrid_score(2.0, 2.0, 0.6) - 1.0) < 1e-9)
    check("hybrid: битые входы не роняют (None→0)",
          rag.hybrid_score(None, None, None) == 0.0)


def test_hybrid_ranking_case():
    section("3. Гибридное ранжирование: специфичный раздел выше общего")
    # Стр. 54 содержит дословно термины запроса; стр. 10 — общее описание,
    # но «векторно» (по фейковому эмбеддингу) может быть близко.
    chunks = [
        ("Общее описание прибора. Назначение, состав, принцип действия, "
         "область применения, характеристики.", {"file": "otc682.pdf", "page": 10}),
        ("ПРОЦЕДУРА Обратная засечка. Координатная засечка станции: "
         "установить прибор, навести на точки, вычислить координаты.",
         {"file": "otc682.pdf", "page": 54}),
    ]
    store = _make_store_with(chunks)
    store.update_settings(hybrid=True, top_k=2, hybrid_alpha=0.6)
    with _PatchEmbed():
        hits = store.search("как выполнить обратную засечку", top_k=2)
    check("гибрид вернул непустой результат", len(hits) == 2)
    check("гибрид: страница 54 (обратная засечка) — первой",
          hits[0]["page"] == 54)
    check("гибрид: у страницы 54 высокий лексический скор",
          hits[0]["lexical"] >= hits[1]["lexical"])

    # Отключаем гибрид — порядок может определяться только косинусом.
    store.update_settings(hybrid=False)
    with _PatchEmbed():
        hits_cos = store.search("как выполнить обратную засечку", top_k=2)
    check("без гибрида лексический скор обнулён",
          all(h["lexical"] == 0.0 for h in hits_cos))


def test_threshold():
    section("4. Фильтрация по порогу min_score")
    chunks = [
        ("обратная засечка координаты станции", {"page": 1}),
        ("совершенно другой текст про погоду", {"page": 2}),
    ]
    store = _make_store_with(chunks)
    store.update_settings(hybrid=True, top_k=5, pool_mult=6, min_score=0.0)
    with _PatchEmbed():
        all_hits = store.search("обратная засечка", top_k=5)
    # Ставим порог выше, чем у «мусорного» чанка — он должен отсечься.
    store.update_settings(min_score=0.9)
    with _PatchEmbed():
        filtered = store.search("обратная засечка", top_k=5)
    check("порог не увеличивает число результатов",
          len(filtered) <= len(all_hits))
    check("порог отсекает нерелевантное (<=1)",
          len(filtered) <= 1)
    if filtered:
        check("после порога все результаты выше min_score",
              all(h["score"] >= 0.9 for h in filtered))


def test_search_multi():
    section("5. search_multi (query rewrite)")
    chunks = [
        ("Обратная засечка (resection) — метод координатной засечки.",
         {"page": 1}),
        ("Настройка яркости дисплея прибора.", {"page": 2}),
    ]
    store = _make_store_with(chunks)
    store.update_settings(hybrid=True, top_k=5, min_score=0.0)
    with _PatchEmbed():
        single = store.search("обратная засечка", top_k=5)
        multi = store.search_multi(["обратная засечка", "resection", "засечка"],
                                   top_k=5)
    check("search_multi вернул результаты", len(multi) >= 1)
    check("search_multi не меньше одного релевантного",
          multi[0]["page"] == 1)
    check("search_multi работает и с пустым списком",
          store.search_multi([], top_k=5) == [])
    check("одна формулировка ≈ search",
          len(single) == len(store.search_multi(["обратная засечка"], top_k=5)))


def test_rerank_callback():
    section("6. Реранкер RagStore.rerank")
    hits = [
        {"page": 1, "text": "a", "score": 0.9},
        {"page": 2, "text": "b", "score": 0.5},
        {"page": 3, "text": "c", "score": 0.3},
    ]
    store = _make_store_with([])

    def fake_rerank(query, hs):
        # Меняем порядок: последний — первым.
        return list(reversed(hs))

    out = store.rerank("q", hits, top_k=3, rerank_fn=fake_rerank)
    check("rerank применил callback (реверс)", out[0]["page"] == 3)
    check("rerank без callback — исходный порядок",
          store.rerank("q", hits, top_k=3)[0]["page"] == 1)
    check("rerank с пустым списком — []", store.rerank("q", []) == [])

    def broken(query, hs):
        raise RuntimeError("boom")

    check("rerank: сбой callback не роняет — исходный порядок",
          store.rerank("q", hits, top_k=3, rerank_fn=broken)[0]["page"] == 1)


def test_settings_roundtrip():
    section("7. Настройки 2-го этапа в RagStore")
    d = tempfile.mkdtemp(prefix="rag_rank_set_")
    idx = os.path.join(d, "idx.db")
    setf = os.path.join(d, "set.json")
    store = rag.RagStore(index_file=idx, settings_file=setf)

    s = store.load_settings()
    check("по умолчанию есть rewrite/hybrid/rerank",
          all(k in s for k in ("rewrite", "hybrid", "hybrid_alpha",
                               "pool_mult", "min_score", "rerank")))

    store.update_settings(rewrite=False, hybrid=False, hybrid_alpha=0.25,
                          pool_mult=10, min_score=0.4, rerank="llm")
    s2 = store.load_settings()
    check("rewrite сохранён", s2["rewrite"] is False)
    check("hybrid сохранён", s2["hybrid"] is False)
    check("hybrid_alpha сохранён", abs(s2["hybrid_alpha"] - 0.25) < 1e-9)
    check("pool_mult сохранён", s2["pool_mult"] == 10)
    check("min_score сохранён", abs(s2["min_score"] - 0.4) < 1e-9)
    check("rerank сохранён", s2["rerank"] == "llm")

    # Клампы.
    store.update_settings(hybrid_alpha=5.0, min_score=9.0, pool_mult=999,
                          rerank="bogus")
    s3 = store.load_settings()
    check("hybrid_alpha клампится ≤1", s3["hybrid_alpha"] <= 1.0)
    check("min_score клампится ≤1", s3["min_score"] <= 1.0)
    check("pool_mult клампится ≤50", s3["pool_mult"] <= 50)
    check("неизвестный rerank игнорируется (остаётся llm)",
          s3["rerank"] == "llm")

    # status() отдаёт новые поля.
    st = store.status()
    check("status содержит rewrite/hybrid/rerank",
          all(k in st for k in ("rewrite", "hybrid", "rerank")))


class _FakeAgent(object):
    """Фейковый агент для rag_llm: без сети, заданные ответы."""

    def __init__(self, text=None, data=None, raise_text=False,
                 raise_json=False):
        self._text = text
        self._data = data
        self._raise_text = raise_text
        self._raise_json = raise_json
        self.calls = []

    def _resolve_model(self, model):
        return ("gigachat", model or "GigaChat")

    def _chat_text(self, provider, model, messages, temperature=None,
                   max_tokens=None):
        self.calls.append(("text", messages))
        if self._raise_text:
            raise RuntimeError("llm down")
        return {"content": self._text or ""}

    def _chat_json(self, provider, model, prompt, temperature=None,
                   max_tokens=None, system=None):
        self.calls.append(("json", prompt))
        if self._raise_json:
            raise RuntimeError("llm down")
        return self._data, ""


def test_llm_rewrite():
    section("8. rag_llm.rewrite_query (фейковый LLM)")
    agent = _FakeAgent(text="- обратная засечка\n2) resection\n  \n* засечка")
    out = rag_llm.rewrite_query(agent, "как сделать обратную засечку", count=3)
    check("rewrite вернул список", isinstance(out, list))
    check("rewrite очистил маркеры списка",
          "обратная засечка" in out and "resection" in out)
    check("rewrite учёл count (<=3)", len(out) <= 3)
    check("rewrite исключил пустые строки", all(x.strip() for x in out))

    check("rewrite: пустой вопрос → []",
          rag_llm.rewrite_query(agent, "  ") == [])
    broken = _FakeAgent(raise_text=True)
    check("rewrite: сбой LLM → [] (graceful)",
          rag_llm.rewrite_query(broken, "вопрос") == [])


def test_llm_rerank():
    section("9. rag_llm.rag_rerank (фейковый LLM)")
    hits = [
        {"file": "a.pdf", "page": 1, "text": "общее описание", "score": 0.9},
        {"file": "a.pdf", "page": 54, "text": "обратная засечка", "score": 0.5},
        {"file": "b.pdf", "page": 3, "text": "погода", "score": 0.4},
    ]
    # LLM ставит лучшую оценку фрагменту id=2 (стр. 54).
    agent = _FakeAgent(data={"scores": [
        {"id": 1, "score": 3}, {"id": 2, "score": 9}, {"id": 3, "score": 1}]})
    out = rag_llm.rag_rerank(agent, "обратная засечка", hits)
    check("rerank вернул все фрагменты", len(out) == 3)
    check("rerank: лучший по LLM — первым (стр. 54)", out[0]["page"] == 54)
    check("rerank: записал поле rerank", "rerank" in out[0])
    check("rerank: отсортировано по убыванию",
          out[0]["rerank"] >= out[1]["rerank"] >= out[2]["rerank"])

    check("rerank: пустой список → []",
          rag_llm.rag_rerank(agent, "q", []) == [])
    noscores = _FakeAgent(data={"scores": []})
    check("rerank: пустые оценки → исходный порядок",
          rag_llm.rag_rerank(noscores, "q", hits)[0]["page"] == 1)
    broken = _FakeAgent(raise_json=True)
    check("rerank: сбой LLM → исходный порядок",
          rag_llm.rag_rerank(broken, "q", hits)[0]["page"] == 1)


def main():
    ensure_utf8()
    test_tokenize()
    test_hybrid_score()
    test_hybrid_ranking_case()
    test_threshold()
    test_search_multi()
    test_rerank_callback()
    test_settings_roundtrip()
    test_llm_rewrite()
    test_llm_rerank()
    return finish("Проверка улучшенного RAG (2-й этап)")


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""Проверка ТРЁХ новых требований к RAG (офлайн, без сети):

  1. ИСТОЧНИК В СКОБКАХ + section/chunk_id:
     * rag.format_citation_id формирует id вида <файл>#стр.P/chunk.C;
     * заголовок фрагмента в build_context содержит id=...;
     * format_sources отдаёт citations (id по каждому фрагменту);
     * Agent._render_sources_html показывает id в скобках.

  2. ЦИТАТЫ ИЗ ИСТОЧНИКА — ЖЁЛТАЯ ПОДСВЕТКА:
     * маркеры [[S]]...[[/S]] → <span class='src-quote'>;
     * запасной путь: «кавычки-ёлочки» тоже подсвечиваются при highlight;
     * «ёлочки» внутри [[S]] не подсвечиваются дважды;
     * без highlight_quotes подсветки нет.

  3. ПОРОГ → «НЕ ЗНАЮ» + ПРОСЬБА УТОЧНИТЬ:
     * RAG_LOW_RELEVANCE_TEXT содержит «не знаю» и просьбу уточнить;
     * Agent.answer(rag_low_relevance=True) возвращает отказ БЕЗ вызова LLM;
     * render_low_relevance_html даёт карточку ответа;
     * сервер выставляет признак низкой релевантности (проверяем
       логику через прямой расчёт на RagStore + rag_context).

Запуск: python -m tests.check_rag_citations
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.harness import check, section, finish, ensure_utf8  # noqa: E402
from rtk_app import rag, html_report  # noqa: E402
from rtk_app.agent import Agent, RAG_LOW_RELEVANCE_TEXT  # noqa: E402


def test_citation_id():
    section("1. Идентификатор ссылки section/chunk_id")
    cid = rag.format_citation_id("/docs/otc682.pdf", 54, 3)
    check("id содержит имя файла", "otc682.pdf" in cid)
    check("id содержит страницу", "стр.54" in cid)
    check("id содержит chunk_id", "chunk.3" in cid)
    check("id без страницы — только chunk",
          rag.format_citation_id("a.pdf", 0, 2) == "a.pdf#chunk.2")
    check("id без чанка — только страница",
          rag.format_citation_id("a.pdf", 7, 0) == "a.pdf#стр.7")
    check("id без страницы и чанка — имя файла",
          rag.format_citation_id("a.pdf", 0, 0) == "a.pdf")
    check("пустой файл — пустая строка",
          rag.format_citation_id("", 1, 1) == "")
    check("берётся basename (без пути)",
          rag.format_citation_id("/x/y/z.pdf", 1, 1).startswith("z.pdf"))


def test_build_context_has_id():
    section("1b. Заголовок фрагмента содержит id")
    # Собираем контекст через реальный RagStore с фейковыми эмбеддингами.
    d = tempfile.mkdtemp(prefix="rag_cit_")
    idx = os.path.join(d, "i.db")
    setf = os.path.join(d, "s.json")
    store = rag.RagStore(index_file=idx, settings_file=setf)

    def fake_embed(text):
        dim = rag.config.RAG_EMBED_DIM
        v = [0.0] * dim
        for ch in str(text).lower():
            v[ord(ch) % dim] += 1.0
        return v

    orig = rag.embed_text
    rag.embed_text = fake_embed
    try:
        conn = store._conn()
        conn.execute(
            "INSERT INTO chunks (id, file, page, chunk, text, vec) "
            "VALUES (?,?,?,?,?,?)",
            (1, "/docs/otc682.pdf", 54, 3, "Обратная засечка выполняется так.",
             rag._pack_vector(fake_embed("Обратная засечка выполняется так."))))
        conn.commit()
        store.update_settings(hybrid=True, top_k=3, min_score=0.0)
        ctx, hits = store.build_context("обратная засечка")
    finally:
        rag.embed_text = orig
    check("контекст непуст", bool(ctx))
    check("в заголовке фрагмента есть id=", "id=" in ctx)
    check("в заголовке есть section/chunk_id (стр.54/chunk.3)",
          "стр.54/chunk.3" in ctx)
    check("hits содержит поле chunk", hits and "chunk" in hits[0])


def test_format_sources_citations():
    section("1c. format_sources отдаёт citations")
    hits = [
        {"file": "/d/a.pdf", "page": 2, "chunk": 1, "score": 0.9},
        {"file": "/d/a.pdf", "page": 5, "chunk": 4, "score": 0.7},
        {"file": "/d/b.pdf", "page": 1, "chunk": 2, "score": 0.8},
    ]
    srcs = rag.format_sources(hits)
    check("два уникальных файла", len(srcs) == 2)
    a = [s for s in srcs if s["file"] == "a.pdf"][0]
    check("у a.pdf два id (стр.2, стр.5)",
          len(a["citations"]) == 2)
    check("id a.pdf содержат chunk", all("chunk." in c for c in a["citations"]))
    check("страницы a.pdf объединены", a["pages"] == [2, 5])

    html = Agent._render_sources_html(srcs)
    check("карточка источников содержит source-cites",
          "source-cites" in html)
    check("в карточке видны id (стр.2/chunk.1)",
          "стр.2/chunk.1" in html)


def test_quote_highlight_markers():
    section("2. Жёлтая подсветка: маркеры [[S]]")
    t = "Факт: [[S]]обратная засечка[[/S]] из документа."
    html, _ = html_report.render_with_memory_counts(t, highlight_quotes=True)
    check("маркер [[S]] превращён в src-quote",
          "src-quote" in html and "обратная засечка" in html)
    check("маркеры [[S]] в выводе не остались",
          "[[S]]" not in html and "[[/S]]" not in html)


def test_quote_highlight_guillemets():
    section("2b. Жёлтая подсветка: «ёлочки» (запасной путь)")
    t = "Согласно документу «обратная засечка» выполняется так."
    html_on, _ = html_report.render_with_memory_counts(
        t, highlight_quotes=True)
    check("ёлочки подсвечены при highlight_quotes",
          "src-quote" in html_on)
    html_off, _ = html_report.render_with_memory_counts(
        t, highlight_quotes=False)
    check("без highlight_quotes подсветки нет",
          "src-quote" not in html_off)


def test_no_double_highlight():
    section("2c. Нет двойной подсветки внутри [[S]]")
    t = "Цитата: [[S]]«вложенная» строка[[/S]] конец."
    html, _ = html_report.render_with_memory_counts(t, highlight_quotes=True)
    check("внутри [[S]] ровно один span",
          html.count("src-quote") == 1)


def test_agent_highlight_wired():
    section("2d. Agent связывает подсветку с наличием RAG")
    # Проверяем через прямой рендер блока с highlight_quotes.
    a = Agent(api_key="dummy")
    single = {"content": "Ответ с «цитатой» из источника.",
              "ok": True, "elapsed": 0.1, "prompt_tokens": 1,
              "completion_tokens": 1}
    block_on, _ = a._render_block("deepseek", "m", "L", "x", single,
                                  highlight_quotes=True)
    check("agent: блок с highlight содержит src-quote",
          "src-quote" in block_on)
    block_off, _ = a._render_block("deepseek", "m", "L", "x", single,
                                   highlight_quotes=False)
    check("agent: блок без highlight не содержит src-quote",
          "src-quote" not in block_off)


def test_low_relevance_text():
    section("3. Текст «не знаю» + просьба уточнить")
    check("содержит «не знаю»", "не знаю" in RAG_LOW_RELEVANCE_TEXT.lower())
    check("просит уточнить", "уточн" in RAG_LOW_RELEVANCE_TEXT.lower())


def test_answer_low_relevance_no_llm():
    section("3b. answer(rag_low_relevance=True) — без вызова LLM")
    calls = []

    class CountingAgent(Agent):
        def _call_one(self, *a, **k):
            calls.append(1)
            return super()._call_one(*a, **k)

    a = CountingAgent(api_key="dummy")
    res = a.answer("нерелевантный вопрос", rag_low_relevance=True)
    check("ответ ok=True", res.get("ok") is True)
    check("LLM НЕ вызывалась", len(calls) == 0)
    check("в ответе есть текст «не знаю»",
          "не знаю" in res.get("text", "").lower())
    check("в html есть карточка rag-low", "rag-low" in res.get("html", ""))
    check("answers помечены low_relevance",
          res["answers"] and res["answers"][0].get("low_relevance") is True)


def test_low_relevance_html():
    section("3c. render_low_relevance_html")
    html = Agent.render_low_relevance_html()
    check("карточка содержит variant-rag-low", "variant-rag-low" in html)
    check("карточка содержит текст «не знаю»",
          "не знаю" in html.lower())
    check("карточка содержит пометку про порог",
          "порог" in html.lower())


def test_threshold_drives_low_relevance():
    section("3d. Порог управляет признаком низкой релевантности")
    # RagStore: высокий порог отсекает все фрагменты -> rag_hits пуст,
    # что на сервере означает rag_low_relevance=True.
    d = tempfile.mkdtemp(prefix="rag_low_")
    store = rag.RagStore(index_file=os.path.join(d, "i.db"),
                         settings_file=os.path.join(d, "s.json"))

    def fake_embed(text):
        dim = rag.config.RAG_EMBED_DIM
        v = [0.0] * dim
        for ch in str(text).lower():
            v[ord(ch) % dim] += 1.0
        return v

    orig = rag.embed_text
    rag.embed_text = fake_embed
    try:
        conn = store._conn()
        conn.execute(
            "INSERT INTO chunks (id, file, page, chunk, text, vec) "
            "VALUES (?,?,?,?,?,?)",
            (1, "d.pdf", 1, 1, "совсем другой текст",
             rag._pack_vector(fake_embed("совсем другой текст"))))
        conn.commit()
        store.update_settings(hybrid=True, top_k=3, min_score=0.99)
        ctx, hits = store.build_context("обратная засечка")
    finally:
        rag.embed_text = orig
    check("при высоком пороге контекст пуст", ctx == "")
    check("при высоком пороге hits пуст -> низкая релевантность", hits == [])

    # И наоборот: низкий порог -> есть hits (релевантность ок).
    rag.embed_text = fake_embed
    try:
        store.update_settings(min_score=0.0)
        ctx2, hits2 = store.build_context("совсем другой текст")
    finally:
        rag.embed_text = orig
    check("при низком пороге фрагменты находятся", len(hits2) >= 1)


def main():
    ensure_utf8()
    test_citation_id()
    test_build_context_has_id()
    test_format_sources_citations()
    test_quote_highlight_markers()
    test_quote_highlight_guillemets()
    test_no_double_highlight()
    test_agent_highlight_wired()
    test_low_relevance_text()
    test_answer_low_relevance_no_llm()
    test_low_relevance_html()
    test_threshold_drives_low_relevance()
    return finish("Проверка источников/цитат/порога RAG")


if __name__ == "__main__":
    sys.exit(main())

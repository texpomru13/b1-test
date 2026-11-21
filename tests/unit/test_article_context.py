from app.retrieval.articles import article_units
from app.retrieval.search import ContextChunk


def sources(texts, order):
    cursor = 0
    chunks = {}
    for i, line in enumerate(texts):
        chunks[i] = ContextChunk(str(i), i, line, None, i + 1, i + 1,
                                 cursor, cursor + len(line), .5)
        cursor += len(line) + 1
    return {str(i): [chunks[i]] for i in order}


def test_article_at_tail_binds_disjoint_paragraphs_with_exact_title_span():
    parts = ["Статья 12\nТребования к стажу\nСтаж равен 10 годам.\nСтатья 13\nВозраст",
             "1. Сведения направляются в комиссию.",
             "2. Кандидатам должно исполниться 35 лет.\nСтатья 14\nИные основания",
             "1. Временный допуск возможен с 30 лет."]
    for order in ([2, 3, 0, 1], [1, 3, 2, 0]):
        units = sources(parts, order)
        eligible = article_units("Какой возраст указан статьёй 13?", units)
        assert eligible == {"0", "1", "2"}
        heading, paragraph = units["2"]
        assert heading.text == "Статья 13\nВозраст"
        assert paragraph.text == "2. Кандидатам должно исполниться 35 лет."
        assert paragraph.list_heading_id == heading.chunk_id
        start, end = paragraph.scope_heading_span
        assert parts[0][start:end] == "Статья 13"
        assert paragraph.page_from == paragraph.page_to == 3


def test_missing_chunk_cannot_prove_scope_even_if_neighbor_uses_same_number():
    parts = ["Статья 13\nВозраст", "1. Первое правило.", "2. Другое правило."]
    units = sources(parts, [0, 2])
    assert article_units("Что определяет статья 13?", units) == {"0"}
    assert not units["2"][0].scope_heading_span


def test_quoted_protocol_heading_and_article_must_be_inside_same_instrument():
    parts = ["Протокол № 7\nСтатья 5\n1. Нормы левого берега.",
             "Хвост первого документа.\nПротокол № 8\nВступительная часть",
             "Статья 5\n1. Нормы правого берега.", "2. Дополнительные сведения.",
             "Статья 6\nДругие нормы.\nПротокол № 11", "Статья 5\n1. Нормы острова."]
    units = sources(parts, [5, 2, 0, 4, 3, 1])
    eligible = article_units("Что в статье 5 Протокола № 8?", units)
    assert eligible == {"2", "3"}
    protocol, article, paragraph = units["3"]
    assert protocol.text == "Протокол № 8"
    assert article.list_heading_id == protocol.chunk_id
    assert paragraph.list_heading_id == article.chunk_id
    assert article.scope_heading_span == (protocol.char_start, protocol.char_end)
    assert article.text.startswith("Статья 5")
    assert paragraph.scope_heading_span == (article.char_start,
                                            article.char_start + len("Статья 5"))
    assert "левого" not in " ".join(c.text for c in units["3"])


def test_mentions_contents_entries_and_explicit_comparison_do_not_create_a_binding():
    parts = ["См. Протокол № 9 в перечне документов.\nСтатья 12 .... 14",
             "1. Согласование производится в течение года."]
    units = sources(parts, [1, 0])
    assert article_units("Что требует статья 12 Протокола № 9?", units) is None
    units = sources(["Статья 6\nОдно условие.", "Статья 8\nИное условие."], [0, 1])
    assert article_units("Сравните статьи 6 и 8.", units) is None


def test_two_adjacent_titles_in_one_source_are_not_quoted_before_the_source_starts():
    parts = ["Предыдущие сведения.\nПротокол № 8\nСтатья 9\nСлужебный порядок", "1. Заявка."]
    units = sources(parts, [1, 0])
    assert article_units("Что устанавливает статья 9 Протокола № 8?", units) == {"0", "1"}
    heading, child = units["1"]
    left, right = child.scope_heading_span
    assert heading.char_start <= left < right <= heading.char_end
    quoted = heading.text[left-heading.char_start:right-heading.char_start]
    assert quoted == "Протокол № 8\nСтатья 9"

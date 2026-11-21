import pytest
from app.retrieval.search import ContextChunk
from app.retrieval.tables import attach_price_headings


def unit(index, text):
    start = index * 200
    return [ContextChunk(str(index), index, text, None, 1, 1, start, start + len(text), .5)]


@pytest.mark.parametrize("reverse", [True, False])
def test_each_numbered_price_row_inherits_only_its_own_header(reverse):
    parts = [unit(2, "Первый зал\nНоменклатура Ед. изм. Цена, руб-"),
             unit(3, "1 Урок рисования шт. 50"), unit(4, "2 Лекция шт. 90"),
             unit(5, "Справка по входу.\nВторой зал\n№ Единица Цена,\nп/п Наименование руб."),
             unit(6, "1 Лекция шт. 130")]
    units = {u[0].chunk_id: u for u in (reversed(parts) if reverse else parts)}
    attach_price_headings(units)
    for row, header in (("4", "2"), ("6", "5")):
        parent, child = units[row]
        assert parent.chunk_id == child.list_heading_id == header
        start, finish = child.table_heading_span
        text = parent.text[start-parent.char_start:finish-parent.char_start]
        assert "Цена," in text and "зал" not in text and "Справка" not in text
    assert units["5"] == parts[3]


@pytest.mark.parametrize("row,index", [("1 Лекция шт. 130", 4), ("7 Лекция шт. 130", 3)])
def test_incomplete_or_restarted_sequence_does_not_inherit_a_currency(row, index):
    units = {"2": unit(2, "№ Наименование Цена, руб."), str(index): unit(index, row)}
    attach_price_headings(units)
    assert len(units[str(index)]) == 1

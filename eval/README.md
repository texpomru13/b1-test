# Проверка качества поиска и ответов

В полном каталоге **27 реальных документов и 323 вопроса**. В него входят 133 вопроса по 14 договорам, 180 вопросов трёх тематических групп и 10 вопросов по сканированному договору о виртуальной АТС. Шесть договоров встречаются в двух сериях вопросов и учитываются один раз при подсчёте документов. [Документы](../test_documents/README.md) · [Все вопросы и эталонные ответы](gold/questions.md) · [Описание сервиса](../README.md) · [Протокол приёмки](../AUDIT.md)

Каждая тематическая группа содержит шесть PDF, 40 вопросов с ответом и 20 вопросов без ответа. Ещё шесть запросов к чужому документу проверяют изоляцию. Полный прогон одной группы охватывает 66 обращений к API.

| Группа | `B1_EVAL_PACK` | Материалы |
|---|---|---|
| Договоры организаций | `contracts-six` | [Вопросы и страницы договоров](packs/contracts-six/source-review.md) |
| Официальные публикации | `published-six` | [Вопросы и страницы публикаций](packs/published-six/source-review.md) · [Первоисточники](packs/published-six/sources.md) |
| Публичные оферты и тарифы | `external-six` | [Вопросы и страницы оферт](packs/external-six/source-review.md) · [Первоисточники](packs/external-six/SOURCES.md) |

В листе для каждого вопроса указаны объект, обязательные факты, необходимые условия и необязательные подробности. Номер страницы у выдержки ведёт к PDF. Для вопроса без ответа проверяющий читает весь документ. Эталон составлен с помощью моделей. Независимую сверку источников фиксирует заполненный человеком лист.

**Приёмка охватывает все 27 документов и 228 HTTP-запросов.** В неё входят все 180 вопросов трёх тематических групп, по два вопроса по каждому из 15 пронумерованных договоров и 18 запросов на изоляцию. Среди 210 обычных вопросов 135 имеют ответ в соответствующем документе, для 75 ожидается отказ. Изоляцию проверяют отдельно, меняя документ. [Зафиксированный список запросов](gold/acceptance-plan.json)

Независимая смысловая оценка относится к группе «Публичные оферты и тарифы» и её 60 вопросам и шести проверкам изоляции. Договоры организаций и официальные публикации участвуют в технической проверке сдаваемой сборки. Материалы по официальным публикациям также использовались для диагностики. Показатели полноты по офертам рассчитываются по её 40 положительным вопросам и размеченным фактам и условиям.

Синтетические контрольные примеры проверяют близкие значения в таблицах и правила с несколькими условиями. Для них доступны 6 PDF, 60 вопросов и 6 запросов на изоляцию. Для запуска выберите `qualification`. [Вопросы и страницы PDF](qualification/source-review.md)

## Стенд

Команды выполняются из корня репозитория с Docker Compose v2 и `uv`. Стенд использует собственные тома и порт. Короткий сценарий `eval/demo.sh` проверяет загрузку, индексацию, список документов, вопросы и изоляцию. Профиль ниже использует локальные модели. В облачном профиле стоимость определяется тарифами провайдера.

```sh
export PROJECT=b1-eval-check
export API=http://127.0.0.1:8021
B1_PROJECT="$PROJECT" ./run --build --provider ollama --quality balanced --strict-profile --port 8021
sh eval/demo.sh "$API"
```

## Проверка всего каталога

Команда `--catalog` загружает 27 документов и подготавливает разметку для всех 323 вопросов в отдельном рабочем каталоге. Экспортируйте индекс того же стенда и запустите вопросы через публичный API. Сырые ответы, manifest и показатели этой серии появятся в `$B1_EVAL_WORK_DIR/runs/catalog/`.

```sh
export B1_EVAL_WORK_DIR="$PWD/eval/work/catalog"
uv run --frozen python eval/upload_corpus.py --catalog --api "$API"
uv run --frozen python eval/export_index.py --project "$PROJECT" --api "$API"
uv run --frozen python eval/catalog.py --check-index
uv run --frozen python eval/run_qa.py --project "$PROJECT" --api "$API" --run catalog
uv run --frozen python eval/metrics.py --run catalog
```

Для оценки смысловой точности одной тематической группы используйте режим `--qualification` ниже. В её отчёте знаменатели рассчитываются по 60 вопросам и шести проверкам изоляции.

### OCR на хосте

Чтобы извлекать текст со сканированных страниц на хосте, нужны `pdftoppm`, Tesseract и языки `rus` и `eng`. В Ubuntu/Debian установите `sudo apt-get install poppler-utils tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng`, в macOS с Homebrew — `brew install poppler tesseract tesseract-lang`. Проверьте языки командой `tesseract --list-langs`. Если обе программы уже установлены, языковые файлы можно скопировать из worker.

```sh
export TESSDATA_PREFIX="$PWD/eval/work/tessdata"
mkdir -p "$TESSDATA_PREFIX"
WORKER="$PROJECT-worker-1"
for lang in rus eng; do
  lang_path=$(docker exec "$WORKER" find /usr/share/tesseract-ocr -name "$lang.traineddata")
  docker cp "$WORKER:$lang_path" "$TESSDATA_PREFIX/$lang.traineddata"
done
tesseract --list-langs
```

## Прогон одной группы

Выберите группу из таблицы и отдельный рабочий каталог. Файлы `lock.json` фиксируют контрольные суммы документов и разметки. `--check-index` проверяет координаты и страницы в каноническом тексте, который использует запущенный сервис. Для независимого замера код, модели, параметры и разметку фиксируют до первого запроса.

```sh
export B1_EVAL_PACK=published-six
export B1_EVAL_WORK_DIR="$PWD/eval/work/acceptance-$B1_EVAL_PACK"
uv run --frozen python eval/qualification_data.py
uv run --frozen python eval/upload_corpus.py --qualification --api "$API"
uv run --frozen python eval/export_index.py --api "$API" --project "$PROJECT"
uv run --frozen python eval/qualification_data.py --check-index
uv run --frozen python eval/run_qa.py --api "$API" --project "$PROJECT" --run final
uv run --frozen python eval/metrics.py --run final
uv run --frozen python eval/qualification_review.py prepare --run final
```

Каталог `$B1_EVAL_WORK_DIR/runs/final/` содержит исходные ответы с HTTP-статусом, данные о сборке, метрики и материалы для проверки. Точность цитат рассчитывается по совпадению текста, документа, фрагмента опубликованного индекса, координат и страниц. Для замера задержек вопросы отправляют последовательно на один API с прогретыми моделями. Судью запускают после измерения.

Команда `uv run --frozen python eval/compile_pack.py "$B1_EVAL_PACK"` проверяет, что данные выбранной группы реальных документов можно собрать из PDF и черновиков. Для собственной копии листа с относительными ссылками используйте `uv run --frozen python eval/qualification_data.py --source-sheet "$B1_EVAL_WORK_DIR/source-review.md"`.

## Оценка содержания

1. До просмотра ответов сверьте разметку с PDF. Укажите имя проверяющего и отметьте согласованные вопросы в листе группы. Оставьте комментарии к спорным случаям.
2. Для оценки ответов человеком откройте `review-input.json` и заполните `claim-review.json` в каталоге прогона. В `reviewer` укажите имя, в `reviewer_type` — `human`. В `required_covered` отметьте каждое обязательное сведение и условие. Для каждого отрывка проверьте все его утверждения, включая дополнения за пределами эталона. Доступны вердикты `supported`, `unsupported`, `wrong_scope`, `contradiction` и `irrelevant`. Для ошибки заполните `reason` и `critical`.
3. Для оценки отдельной локальной моделью Ollama выполните `uv run --frozen python eval/qualification_review.py model --run final`. По умолчанию используется `qwen3.6:27b-q4_K_M`. Если нужны оценки и модели, и человека, сохраните их в отдельных копиях каталога прогона.
4. Рассчитайте результат командой `uv run --frozen python eval/qualification_review.py score --run final --source-review ПУТЬ_К_ЛИСТУ.md`. В `qualification.json` будут доли, ошибки, результаты по каждому порогу, тип проверяющего и контрольные суммы. Пустой лист оставляет сверку источников открытой.

| Показатель каждой группы | Порог |
|---|---|
| Покрытие обязательных фактов и условий | ≥ 90% |
| Полные корректные ответы среди 40 вопросов с ответом в документе | ≥ 34/40 |
| Критические смысловые ошибки | 0 |
| Верные отказы на вопросах без ответа | 20/20 |
| Верные отказы при запросе к чужому документу | 6/6 |
| Валидность ответов и точность координат цитат | 100% |

Выданный ответ с ложным дополнением не засчитывается как полный, даже если содержит необходимые факты. Отказ или техническая ошибка на вопросе с ответом в документе дают нулевое покрытие его фактов. Показатель `found` описывает решение выдать ответ. Смысловую точность и область действия утверждения оценивают отдельно.

## Подробная диагностика поиска на договорах

В серии на 14 документах 133 вопроса, из них 93 с ответом и 40 без ответа. Дополнительно задаются 17 вопросов к чужому документу. Разделение по документам даёт 35 вопросов в dev и 98 в holdout. Часть примеров служит для диагностики и проверки регрессий. Файл на 100 страниц и скан № 06 проверяют индексацию. [Вопросы и эталонные выдержки](gold/gold.jsonl) · [Документы](../test_documents)

```sh
export B1_EVAL_WORK_DIR="$PWD/eval/work/verification"
uv run --frozen python eval/upload_corpus.py --api "$API"
uv run --frozen python eval/export_index.py --project "$PROJECT" --api "$API"
uv run --frozen python eval/build_gold.py
uv run --frozen python eval/run_qa.py --api "$API" --project "$PROJECT" --run balanced-check
for split in all dev holdout; do
  uv run --frozen python eval/metrics.py --run balanced-check --split "$split"
done
uv run --frozen python eval/report.py --runs balanced-check
```

Загрузка возобновляет опрос поставленных задач. Скрипт `build_gold.py` находит координаты цитат в экспортированном тексте и записывает `$B1_EVAL_WORK_DIR/gold.jsonl`. При отсутствии рабочего файла оценщик использует размеченный корпус из репозитория. `metrics.py` проверяет структуру ответа и наличие доказательств у кандидатов, в контексте и в итоговых цитатах. Вопрос `03-06` требует одновременно вводного запрета и подпунктов. [Пример обоснования запрета](reports/submission/governing-clauses.json)

Для этой серии `judge.py` с отдельной локальной моделью считает полноту по перечисленным фактам и появление перечисленных запрещённых утверждений. Снимите флаг `--skip-ragas` для метрик RAGAS. Пока стенд работает, можно сравнить dense, BM25, RRF, DBSF и RRF с реранкером на одних и тех же фрагментах.

```sh
uv run --frozen --project eval python eval/judge.py --run balanced-check --skip-ragas
uv run --frozen python eval/report.py --runs balanced-check
uv run --frozen python eval/ablation.py --project "$PROJECT" --name balanced
B1_PROJECT="$PROJECT" ./run stop
```

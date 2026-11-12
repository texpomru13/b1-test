# Document QA RAG backend

Backend-сервис вопросов по документам (русскоязычные договоры, PDF/DOCX до 100
страниц). Принимает документ, фоново индексирует его в Qdrant и отвечает на
вопросы **только по тексту документа**, возвращая ответ со ссылками на конкретные
места. Если сведений в документе нет — честно возвращает `found=false`.

Стек: FastAPI · Celery + Redis · Qdrant (одна коллекция, dense + BM25 в каждой
точке, объединение одним запросом) · обязательный реранкер · LangChain
(разбиение, вызовы LLM) · LangGraph (конечный QA-граф без памяти) · Pydantic.
Пользовательского интерфейса нет.

## Содержание

1. [Быстрый старт](#1-быстрый-старт)
2. [Работа с `./run`](#2-работа-с-run)
3. [API: маршруты и JSON-контракты](#3-api-маршруты-и-json-контракты)
4. [Примеры запросов с результатами](#4-примеры-запросов-с-результатами)
5. [Конфигурация](#5-конфигурация)
6. [Архитектура](#6-архитектура)
7. [Принятые решения](#7-принятые-решения)
8. [Доработки для production](#8-доработки-для-production)
9. [Что проверено](#9-что-проверено)
10. [Соответствие заданию и известные ограничения](#10-соответствие-заданию-и-известные-ограничения)
11. [Разработка и тесты](#11-разработка-и-тесты)

---

## 1. Быстрый старт

Нужен только установленный и запущенный Docker Engine/Desktop с Compose v2
(Windows — подготовленный WSL2 с Docker integration). Ни Python, ни ручного
`.env` на хосте не требуется.

```bash
./run            # выбор профиля → подготовка моделей → старт → реальная готовность
```

Когда launcher напечатает `Ready.`, API доступен на <http://127.0.0.1:8000>
(Swagger UI — `/docs`, готовность — `/health/ready`).

```bash
# 1. загрузить документ (DOCX требует явного type=, PDF — application/pdf)
curl -X POST http://127.0.0.1:8000/api/v1/documents \
  -F 'title=Договор поставки' \
  -F 'file=@contract.pdf;type=application/pdf'

# 2. дождаться status=succeeded
curl http://127.0.0.1:8000/api/v1/jobs/<job_id>

# 3. задать вопрос
curl -X POST http://127.0.0.1:8000/api/v1/documents/<doc_id>/questions \
  -H 'Content-Type: application/json' \
  -d '{"question": "Какая неустойка предусмотрена пунктом 4.2?", "debug": true}'
```

Остановить: `./run stop`. Данные и скачанные модели сохраняются в именованных
volumes и при следующем `./run` не перекачиваются.

Первый запуск скачивает десятки ГиБ закреплённых весов (профиль `quality`:
энкодер Giga-Embeddings-instruct 3B, реранкер Qwen3-Reranker-8B, LLM
Qwen3.6-27B Q4 ≈ 17 ГБ) — см. [ресурсы](#ресурсы).

---

## 2. Работа с `./run`

`./run` — POSIX-shell launcher в корне репозитория. Работает из любого каталога
(корень определяется по пути скрипта), при повреждённом executable-bit —
`sh ./run`. Профиль подбирает чистый резолвер `scripts/resolver.py`, который
исполняется внутри закреплённого python-образа; факты о хосте собирает
`scripts/probe.sh`.

### Команды

| Команда | Что делает |
|---|---|
| `./run` или `./run up` | Полный запуск: профиль → модели → `docker compose up -d --build` → ожидание реальной готовности → итоговая сводка |
| `./run doctor` | Диагностика без запуска контейнеров стека и без загрузки моделей: версии Docker/Compose, факты хоста (ОС, память хоста и Docker VM, доступность host Ollama / DMR / NVIDIA, свободный диск) и решение резолвера с причинами |
| `./run doctor --json` | То же в JSON `{"facts": …, "resolution": …}`. Первая строка вывода — служебная (`==> [preflight] …`), для разбора отбросьте её: `./run doctor --json \| tail -n +2` |
| `./run --dry-run` | Показывает решение резолвера и итоговые переменные; контейнеры не стартуют, модели не скачиваются. Внимание: решение записывается в `.runtime/` и учитывается следующим запуском |
| `./run status` | `docker compose ps` + фактическое тело `/health/ready` |
| `./run logs` | Последние 200 строк логов всех сервисов. Для потока: `docker compose -p b1-rag logs -f api` |
| `./run stop` | Останавливает контейнеры проекта |
| `./run down` | Удаляет контейнеры и сеть проекта. **Volumes сохраняются всегда** (индекс, документы, модели) |

Удаление данных launcher не выполняет никогда. Осознанный полный сброс делается
вручную: `docker compose -p b1-rag down -v` (удалит индекс Qdrant, Redis,
загруженные документы и кэш моделей).

### Флаги

| Флаг | Значения | Действие |
|---|---|---|
| `--provider` | `ollama` \| `dmr` \| `openai` \| `anthropic` | Явный выбор LLM-runtime. Значение не валидируется: опечатка трактуется как автовыбор |
| `--quality` | `quality` \| `quality-compact` \| `balanced` \| `low-memory` | Стартовый профиль качества (см. таблицу ниже) |
| `--model` | id модели | LLM. Для Ollama — тег (`qwen3.6:27b-q4_K_M`), для DMR — id DMR (`ai/qwen3.6:27B`), для облака — id модели провайдера |
| `--embedding-model` | `ai-sage/Giga-Embeddings-instruct` \| `ai-sage/Giga-Embeddings-instruct-480M-0826` | Энкодер. Смена энкодера на существующем индексе несовместима с его fingerprint — индексация вернёт `INDEX_FINGERPRINT_MISMATCH`, данные не удаляются |
| `--reranker-model` | `Qwen/Qwen3-Reranker-8B` \| `BAAI/bge-reranker-v2-m3` | Реранкер. Переиндексация не нужна |
| `--strict-profile` | — | При нехватке памяти — ошибка вместо перехода на меньший профиль |
| `--device` | `auto` \| `cpu` \| `metal` \| `nvidia` | Сейчас действует только `nvidia`: без успешного контейнерного GPU-smoke запуск отклоняется (скрытой подмены на CPU нет). Остальные значения на выбор не влияют |
| `--port` | номер | Порт API на `127.0.0.1` (по умолчанию 8000). Занятый чужим процессом порт — ошибка с подсказкой; чужие процессы не завершаются |
| `--offline` | — | Запуск только из подготовленных образов и моделей, сеть контейнеров становится internal (см. [Offline](#offline)) |
| `--build` | — | Принимается; сборка образов из исходников выполняется при каждом `up` (слои кэшируются) |
| `--residency`, `--allow-slow` | — | Принимаются, но на поведение пока не влияют |

Выбор моделей и профиля можно задать и переменными окружения (в shell или в
`.env`): `LLM_PROVIDER`, `LLM_MODEL`, `EMBEDDING_MODEL`, `RERANKER_MODEL`,
`QUALITY_PROFILE`, `PROFILE_STRICT`, `DEVICE`, `OFFLINE` действуют так же, как
одноимённые флаги. Приоритет: флаг CLI → окружение или `.env` → сохранённый
профиль → автовыбор; значение `auto` означает «не задано».

```bash
RERANKER_MODEL=BAAI/bge-reranker-v2-m3 ./run     # то же, что --reranker-model
FUSION_METHOD=dbsf CONTEXT_TOP_K=5 ./run         # параметры поиска, см. раздел 5
```

Между запусками сохраняются только выбранный локальный runtime и энкодер
(`.runtime/resolved.env`). `--quality`, `--model`, `--reranker-model`, `--port`,
`--strict-profile` действуют на один запуск — повторяйте их при каждом `./run`.
Сохранённый runtime **не мигрирует молча**: если после первого запуска появился
более быстрый runtime (например, установлена host Ollama), launcher напишет
`saved profile kept…` и оставит прежний; переключение — только явным
`--provider`.

### Стадии запуска

```
[profile]   проба хоста и выбор профиля (один раз за запуск), причины решения
[download]  host Ollama: ollama pull модели (только профиль host-ollama)
[images]    сборка образов и docker compose up -d
[download]  ожидание models-init (закреплённые ревизии энкодера и реранкера → volume hf_models)
[download]  ожидание ollama-init (только контейнерный Ollama)
[warmup]    ожидание загрузки энкодера и реранкера в local-models
[services]  ожидание liveness API
[ready]     ожидание полной готовности /health/ready
```

Фактический вывод успешного повторного запуска (Mac, host Ollama, всё из кэша;
лог сборки образов сокращён):

```
==> [profile] probing host facts and resolving the profile (once per launch)
    - provider=ollama: verified host Ollama endpoint
    - resolved profile=quality runtime=host-ollama encoder=ai-sage/Giga-Embeddings-instruct reranker=Qwen/Qwen3-Reranker-8B residency=resident-both
==> [download] ensuring host Ollama has qwen3.6:27b-q4_K_M (resumable pull)
==> [images] building/starting services (first run may download many GiB)
==> [download] waiting for models-init (pinned snapshots into the hf_models volume)
==> [warmup] waiting for local-models (encoder + reranker load)
==> [services] waiting for API liveness
==> [ready] waiting for full readiness

Ready. Resolved stack (candidate preset, reasons above):
  provider=ollama model=qwen3.6:27b-q4_K_M
  encoder=ai-sage/Giga-Embeddings-instruct@2cf0fdc9
  reranker=Qwen/Qwen3-Reranker-8B (qwen_yes_no)
  API: http://127.0.0.1:8000  (docs: /docs, ready: /health/ready)
  Logs: ./run logs   Status: ./run status   Stop: ./run stop   Down (keep data): ./run down
```

Ожидания конечны: models-init — 30 мин, ollama-init — 40 мин, прогрев
local-models — 15 мин, liveness API — 5 мин, полная готовность — 10 мин
(меняется переменной `RUN_READY_TIMEOUT_POLLS`, шаг 5 с). При таймауте launcher
печатает фактическое состояние компонентов и никогда не пишет `Ready`.

### Выбор LLM-runtime

Поиск (энкодер, BM25, реранкер) во всех режимах работает локально в контейнере
`local-models`. Выбирается только, где исполняется LLM:

| Runtime | Когда выбирается | Compose-файлы |
|---|---|---|
| host Ollama (Metal) | macOS: автоматически, если на `127.0.0.1:11434` отвечает host Ollama; либо `--provider ollama` | `compose.yaml` + `compose.mac.yaml` |
| контейнерный Ollama (CPU) | автоматически, если ускоренного runtime нет; базовый путь Linux | `compose.yaml` |
| Docker Model Runner | только явно: `--provider dmr` (автопредпочтение закрыто гейтом совместимости, ADR 005) | `compose.yaml` + `compose.dmr.yaml` |
| контейнерный Ollama + NVIDIA | только после успешного контейнерного GPU-smoke | `compose.yaml` + `compose.gpu.yaml` |
| OpenAI / Anthropic | только явно: `--provider openai\|anthropic` + ключ | `compose.yaml` + `compose.cloud.yaml` |

Для `--provider dmr` модель нужно заранее загрузить и указать её DMR-идентификатор:

```bash
docker model pull ai/qwen3.6:27B
./run --provider dmr --model ai/qwen3.6:27B
```

### Профили качества

Резолвер начинает с запрошенного профиля (по умолчанию `quality`) и переходит к
следующему **только при измеренной нехватке памяти**, печатая
`!! QUALITY DEGRADED: <причина>`. CPU сам по себе качество не снижает.

| Профиль | Энкодер | Реранкер | LLM | Оценка памяти резолвера, ГиБ (энкодер + реранкер + LLM) |
|---|---|---|---|---|
| `quality` | Giga-Embeddings-instruct (3B, dim 2048) | Qwen3-Reranker-8B | qwen3.6:27b-q4_K_M | 14.5 + 19.0 + 20.0 |
| `quality-compact` | Giga-Embeddings-instruct | bge-reranker-v2-m3 | qwen3.6:27b-q4_K_M | 14.5 + 2.2 + 20.0 |
| `balanced` | Giga-Embeddings-instruct-480M (dim 1024) | bge-reranker-v2-m3 | qwen3.5:9b-q4_K_M | 3.6 + 2.2 + 7.5 |
| `low-memory` | Giga-Embeddings-instruct-480M | bge-reranker-v2-m3 | qwen3.5:4b-q4_K_M | 3.6 + 2.2 + 3.5 |

Все пресеты имеют статус `candidate`. Профили `balanced` и `low-memory` прошли
оценку на gold-наборе этапа 9 (см. раздел 9 и `eval/reports/REPORT.md`), но набор
составлен агентом и человеком не ревьюился, поэтому статус не повышен до `validated`;
`quality` / `quality-compact` на gold-наборе не оценивались. Дедлайны зависят от runtime: host Ollama / DMR / GPU — чтение LLM
300 с, вопрос целиком 900 с; облако — 300 / 420 с; контейнерный CPU — 1200 /
3600 с.

### Облачные провайдеры

Облако включается только явным `--provider`; наличие ключа само по себе ничего
не включает, скрытого перехода local → cloud нет. Ключ берётся из окружения или
из необязательного `.env` в корне и не попадает в аргументы и в `.runtime/`.

```bash
OPENAI_API_KEY=sk-...       ./run --provider openai    --model <id модели>
ANTHROPIC_API_KEY=sk-ant-... ./run --provider anthropic --model <id модели>
```

`--model` для облака обязателен, оба флага нужно передавать при каждом запуске.
Локальная LLM при этом не скачивается и не стартует; энкодер и реранкер остаются
локальными. Живые вызовы облачных провайдеров не выполнялись (`not_run`: ключей
нет, платные прогоны выключены).

### Offline

`./run --offline` стартует только из уже подготовленных образов и моделей:
init-задачи ничего не скачивают и перечисляют недостающее, сеть compose
становится internal — исходящий трафик контейнеров реально заблокирован. В этом
режиме недоступны host Ollama / DMR / облако (используйте контейнерный профиль),
а порт API на хост не публикуется — обращаться к сервису нужно изнутри сети,
например `docker compose -p b1-rag exec api python …`.

### Файлы launcher

| Путь | Назначение |
|---|---|
| `.runtime/resolved.env` | Итоговые переменные `B1_*` для compose; пишется атомарно до интерполяции |
| `.runtime/resolved-config.json` | Полное решение резолвера с причинами |
| `.runtime/launcher.lock` | Блокировка от параллельных `./run`. Если launcher был убит (SIGKILL, отключение питания) — удалите каталог вручную |
| `.env` | Необязательный файл настроек (шаблон — `.env.example`). Читается `./run` и `docker compose`; launcher его никогда не перезаписывает |

### Запуск без launcher

```bash
docker compose up --build
```

Поднимает контейнерный CPU-стек с умолчаниями из `compose.yaml` (профиль
`quality`, контейнерный Ollama). Подбора профиля под память здесь нет: все три
модели загружаются как есть, поэтому Docker VM нужно выделить достаточно памяти.
Модели можно переопределить переменными `B1_*` (см. [конфигурацию](#5-конфигурация)).

### Windows

`.\run.ps1` передаёт аргументы в `./run` внутри подготовленного WSL2-дистрибутива
с Docker integration; WSL сам не устанавливается. Статус: написан по
спецификации, живой запуск `not_run` (нет Windows-машины).

### Коды завершения

| Код | Причина |
|---|---|
| 0 | Успех |
| 2 | Неизвестная опция или команда |
| 3 | Резолвер отклонил конфигурацию (нехватка памяти при `--strict-profile`, неизвестный профиль или модель, `--device nvidia` без GPU-smoke, DMR не запущен) |
| 4 | Не удалось подготовить модели (`ollama pull`, models-init, ollama-init, отсутствие модели в offline) |
| 10 / 11 / 12 | Docker не установлен / демон недоступен / нет плагина Compose v2 |
| 13 | Порт занят чужим процессом (`./run --port 8001`) |
| 14 | Уже выполняется другой `./run` |
| 20 | local-models или API не стали healthy |
| 21 | Таймаут полной готовности |
| 130 | Прервано (Ctrl-C); состояние контейнеров и volumes сохраняется, повторный `./run` продолжит |

Сбой самого `docker compose up` (например, обрыв сети при сборке) возвращает код
compose. `doctor`, `status`, `logs` всегда завершаются с кодом 0.

### Ресурсы

- Память: профиль `quality` держит в контейнере `local-models` оба веса поиска
  одновременно (в журнале зафиксировано 46.7 ГБ RSS на Docker VM 63 ГиБ);
  LLM 27B Q4 занимает ещё ≈ 20 ГиБ — в памяти хоста (host Ollama / DMR) или
  в той же Docker VM (контейнерный Ollama). На машинах меньше резолвер переходит
  на младший профиль с явным предупреждением.
- Диск: веса моделей — десятки ГиБ в volumes `hf_models` и `ollama_models`
  (или в хранилище host Ollama).
- Если на хосте есть кэш `~/.cache/huggingface`, models-init копирует закреплённые
  ревизии из него (монтируется только на чтение) вместо загрузки.
- Время ответа на вопрос на локальных моделях — от десятков секунд до минут;
  измеренные значения — в [разделе 4](#4-примеры-запросов-с-результатами).

---

## 3. API: маршруты и JSON-контракты

Базовый адрес: `http://127.0.0.1:8000`. Бизнес-маршруты — под `/api/v1`. Тела
запросов и ответов — JSON (кроме multipart-загрузки и выдачи PDF).
Аутентификации нет (см. [production](#8-доработки-для-production)).
`doc_id` и `job_id` — 32 hex-символа. Время — ISO-8601 UTC.

### Маршруты

| Метод | Путь | Операция задания | Успех |
|---|---|---|---|
| `POST` | `/api/v1/documents` | загрузка документа и запуск индексации | 202 |
| `GET` | `/api/v1/jobs/{job_id}` | состояние фоновой задачи индексации | 200 |
| `GET` | `/api/v1/documents` | список загруженных документов | 200 |
| `POST` | `/api/v1/documents/{doc_id}/questions` | ответ на вопрос по документу; `debug: true` добавляет диагностику | 200 |
| `GET` | `/api/v1/documents/{doc_id}` | карточка документа | 200 |
| `GET` | `/api/v1/documents/{doc_id}/normalized` | PDF, к страницам которого относятся цитаты | 200 |
| `GET` | `/health/live`, `/health/ready` | liveness / readiness | 200, 200 или 503 |
| `GET` | `/docs`, `/redoc`, `/openapi.json` | OpenAPI | 200 |

Заголовки: `X-Request-ID` принимается от клиента или генерируется, возвращается
в каждом ответе и попадает в `error.request_id`.

### 3.1 `POST /api/v1/documents` — загрузка и запуск индексации

Запрос `multipart/form-data`:

| Где | Имя | Обяз. | Описание |
|---|---|---|---|
| form | `file` | да | PDF (`application/pdf`) или DOCX (`application/vnd.openxmlformats-officedocument.wordprocessingml.document`). Проверяются заявленный Content-Type части и сигнатура файла; расширение имени не учитывается |
| form | `title` | нет | Название документа, до 500 символов |
| header | `Idempotency-Key` | нет | Ключ идемпотентности. Без заголовка ключ вычисляется из содержимого файла, `title` и настроек разбиения |

Лимиты: размер ≤ `MAX_FILE_MB` (25 МиБ) — проверяется при загрузке; страниц ≤
`MAX_PAGES` (100) — проверяется в фоновой задаче (для DOCX — после конвертации в
PDF), превышение видно в статусе задачи как `failed` / `PAGE_LIMIT_EXCEEDED`.
Сканы без текстового слоя и PDF с паролем принимаются с 202 и завершаются
`failed` с кодами `OCR_REQUIRED` / `PASSWORD_PROTECTED`.

Идемпотентность: тот же ключ и тот же файл → 202 с прежними `doc_id`/`job_id`
(повторной индексации нет); тот же ключ с другим файлом, названием или
настройками → 409.

Ответ **202** `UploadAccepted` + заголовок `Location: <status_url>`:

| Поле | Тип | Описание |
|---|---|---|
| `doc_id` | string | Идентификатор документа |
| `job_id` | string | Идентификатор задачи индексации |
| `status` | string | `accepted` |
| `status_url` | string | `/api/v1/jobs/{job_id}` |
| `document_url` | string | `/api/v1/documents/{doc_id}` |

Ошибки: 409 `CONFLICT`, 413 `PAYLOAD_TOO_LARGE`, 415 `UNSUPPORTED_MEDIA_TYPE`
(тип, сигнатура, пустой файл), 422 `VALIDATION_ERROR`, 503
`DEPENDENCY_UNAVAILABLE` (Redis или файловое хранилище).

### 3.2 `GET /api/v1/jobs/{job_id}` — состояние задачи

Состояние читается из собственного реестра задач в Redis. Ответ **200**
`JobResponse`:

| Поле | Тип | Описание |
|---|---|---|
| `job_id`, `doc_id` | string | |
| `status` | string | `accepted` → `queued` → `running` → `succeeded` \| `failed`; промежуточные `retry_wait` (повторная постановка после сбоя воркера), `compensating`, `cleanup_pending` (откат незавершённой индексации). Терминальные: `succeeded`, `failed` |
| `stage` | string | `upload`, `convert` (DOCX), `parse`, `normalize`, `split`, `embed`, `upsert`, `verify`, `publish` |
| `attempt` | integer | Номер попытки выполнения |
| `progress` | object | `{"completed": int, "total": int \| null, "unit": "chunks"}` |
| `created_at`, `updated_at` | string | |
| `error` | object \| null | `{"code": string, "message": string, "retryable": bool}` при сбое |

Коды `error.code`: `PAGE_LIMIT_EXCEEDED`, `PASSWORD_PROTECTED`, `OCR_REQUIRED`,
`PARSE_FAILED`, `CONVERSION_FAILED`, `CONVERSION_TIMEOUT`, `EMBEDDING_UNAVAILABLE`,
`QDRANT_UNAVAILABLE`, `INDEX_SCHEMA_MISMATCH`, `INDEX_FINGERPRINT_MISMATCH`,
`INDEX_VERIFY_FAILED`, `PUBLISH_FAILED`, `ATTEMPTS_EXHAUSTED`. Неизвестный `job_id` → 404 `NOT_FOUND`.

### 3.3 `GET /api/v1/documents` — список документов

Query: `limit` (1…100, по умолчанию 20), `cursor` (значение `next_cursor`
предыдущей страницы). Порядок — от новых к старым. Ответ **200** `DocumentList`:

| Поле | Тип | Описание |
|---|---|---|
| `items[].doc_id` | string | |
| `items[].title` | string \| null | |
| `items[].status` | string | Статус задачи индексации документа |
| `items[].ready` | boolean | `true`, когда документу можно задавать вопросы |
| `items[].job_id` | string | |
| `items[].created_at` | string | |
| `next_cursor` | string \| null | `null` на последней странице |

### 3.4 `GET /api/v1/documents/{doc_id}` — карточка документа

Ответ **200** `DocumentDetail`: поля элемента списка плюс `original_filename`,
`content_type`, `size_bytes`, `updated_at`, `page_count`, `chunk_count`
(последние два появляются после разбора). Неизвестный документ → 404.

### 3.5 `GET /api/v1/documents/{doc_id}/normalized`

Отдаёт PDF (`application/pdf`), к которому относятся номера страниц в цитатах:
для DOCX — результат серверной конвертации, для PDF — исходный файл. 404, если
документ неизвестен или PDF ещё не готов.

### 3.6 `POST /api/v1/documents/{doc_id}/questions` — ответ на вопрос

Каждый вопрос обрабатывается независимо: истории, сессий и тредов нет. Запрос
синхронный и на локальных моделях длится минуты — задавайте клиенту таймаут не
меньше `QA_DEADLINE_S` (измеренное время на проверенной конфигурации — 2–6
минут, см. [примеры](#4-примеры-запросов-с-результатами)).

Тело `QuestionRequest`:

| Поле | Тип | Обяз. | Описание |
|---|---|---|---|
| `question` | string | да | От 1 до `MAX_QUESTION_CHARS` символов (по умолчанию 4000) |
| `debug` | boolean | нет (`false`) | **Необязательный диагностический параметр**: `true` добавляет в ответ блок `diagnostics` |

Любые другие поля (`history`, `thread_id`, `messages` …) отклоняются с 422.

Ответ **200** `AnswerResponse`:

| Поле | Тип | Описание |
|---|---|---|
| `answer` | string | При `found=true` — утверждения с маркерами `[n]`, ссылающимися на `citations[].id`. При `found=false` — «В документе не найдено достаточно сведений для ответа на этот вопрос.» |
| `found` | boolean | |
| `citations` | array | При `found=false` — пустой список |
| `diagnostics` | object | Только при `debug=true` |

`Citation`:

| Поле | Тип | Описание |
|---|---|---|
| `text` | string | Дословная цитата — точная подстрока текста документа |
| `page` | integer | Страница начала цитаты (с 1) |
| `section` | string | Раздел документа; ключ отсутствует, если фрагмент находится до первого нумерованного заголовка |
| `id` | integer | Номер, на который ссылается маркер `[n]` в `answer` |
| `page_to` | integer | Страница конца цитаты |
| `chunk_id` | string | Идентификатор фрагмента (точки Qdrant) |
| `char_start`, `char_end` | integer | Смещения цитаты в каноническом тексте документа |

Поля `text`, `page`, `section` совпадают с примером из задания; остальные —
дополнения. Ответ собирает сервер: модель возвращает утверждения со ссылками на
фрагменты, сервер проверяет, что каждая цитата дословно присутствует в выданном
модели фрагменте и что числа утверждения есть в цитатах, восстанавливает
страницы и смещения, затем второй вызов модели проверяет, что утверждения
подтверждаются цитатами. `found=true` возвращается только если все утверждения
подтверждены.

`diagnostics` (при `debug=true`):

| Требование задания | Поле | Описание |
|---|---|---|
| сколько фрагментов дала каждая ветвь поиска | `dense_returned`, `sparse_returned` | Число фрагментов плотной ветви и ветви BM25. Считаются отдельным диагностическим повтором запроса (`branch_counts_source = "diagnostic_replay"`) и отражают реальное число, а не лимит. При сбое повтора ключи отсутствуют, причина — в `diagnostic_warnings` |
| сколько кандидатов осталось после переранжирования | `context_count` | Число фрагментов, оставшихся после отсечения по `CONTEXT_TOP_K` и бюджету контекста и переданных LLM |
| | `fused_returned`, `reranked_count` | Число кандидатов объединённого списка и число кандидатов, оценённых реранкером |
| какие из них использованы при формировании ответа | `context_chunk_ids` | Фрагменты, переданные в контекст LLM |
| | `cited_chunk_ids` | Фрагменты, на которые опираются цитаты ответа (подмножество `context_chunk_ids`) |
| — | `rerank_scores` | `{chunk_id: оценка реранкера}` для всех оценённых кандидатов |
| — | `stage_ms` | Длительности `encode`, `fusion`, `rerank`, `context`, мс |
| — | `model_ids` | Фактические `embedding`, `reranker`, `llm` |
| — | `token_usage` | `input_tokens`, `output_tokens`, `llm_calls` |

Исходы и коды:

| Код | `error.code` | Когда | `retryable` |
|---|---|---|---|
| 200 | — | `found=true` либо `found=false` (в документе недостаточно сведений) | — |
| 404 | `NOT_FOUND` | Документ неизвестен | нет |
| 409 | `CONFLICT` | Документ ещё индексируется (повторить позже) или индексация завершилась `failed` | да / нет |
| 422 | `VALIDATION_ERROR` | Невалидное тело | нет |
| 429 | `RATE_LIMITED` | Уже выполняется `MAX_CONCURRENT_QA` вопросов (по умолчанию 1 на весь сервис) | да |
| 502 | `MODEL_OUTPUT_INVALID` | Модель не выдала валидный структурированный ответ | да |
| 503 | `DEPENDENCY_UNAVAILABLE` | Недоступны эмбеддинги, Qdrant, реранкер или LLM. Недоступность реранкера — ошибка, а не ответ без переранжирования | да |
| 504 | `DEADLINE_EXCEEDED` | Превышен `QA_DEADLINE_S` | да |

Сбои инфраструктуры не маскируются под `found=false`.

### 3.7 Формат ошибок

Все ошибки API возвращаются в едином конверте:

```json
{"error": {"code": "NOT_FOUND", "message": "…", "retryable": false, "request_id": "…"}}
```

Коды: `VALIDATION_ERROR` (422), `NOT_FOUND` (404), `CONFLICT` (409),
`PAYLOAD_TOO_LARGE` (413), `UNSUPPORTED_MEDIA_TYPE` (415), `RATE_LIMITED` (429),
`MODEL_OUTPUT_INVALID` (502), `DEPENDENCY_UNAVAILABLE` (503),
`DEADLINE_EXCEEDED` (504), `INTERNAL` (500).

### 3.8 Health

- `GET /health/live` → 200 `{"status": "alive"}`.
- `GET /health/ready` → 200, если доступны Redis, Qdrant и сервис локальных
  моделей; иначе 503 с тем же телом:
  `{"status": "ready"|"not_ready", "components": {"redis": "ok", "qdrant": "ok", "local_models": "ok"}, "model_ready": true}`.
  LLM-runtime в readiness не входит: его недоступность проявится ответом 503 на вопрос.

---

## 4. Примеры запросов с результатами

Все ответы ниже — фактические результаты прогона 2026-11-11 на стеке,
поднятом командой `./run`: macOS Apple Silicon (M4 Max), профиль `quality`,
LLM qwen3.6:27b-q4_K_M в host Ollama (Metal), энкодер Giga-Embeddings-instruct
и реранкер Qwen3-Reranker-8B в контейнере на CPU. Вывод отформатирован; в первом
примере сокращён только словарь `rerank_scores`.

Тестовый документ — синтетический договор поставки (DOCX, 1 страница, 13
фрагментов): преамбула со сторонами, разделы 1–5 с пунктами 1.1–5.1, в том числе
«2.1 Общая цена договора составляет 4 850 000 … рублей» и «4.2 За нарушение
сроков поставки … неустойку в размере 0,1% …».

### 4.1 Загрузка документа

```bash
curl -i -X POST http://127.0.0.1:8000/api/v1/documents \
  -H 'Idempotency-Key: readme-demo-1' \
  -F 'title=Договор поставки № ДП-2026/117' \
  -F 'file=@contract.docx;type=application/vnd.openxmlformats-officedocument.wordprocessingml.document'
```

```
HTTP/1.1 202 Accepted
location: /api/v1/jobs/fad1c49b91fd4e9f880bb74bc7a65d8c
x-request-id: 64892b8e51064bf7a2983da7479bfd0c
```

```json
{
  "doc_id": "fa0526d673424b3490680750290545f0",
  "job_id": "fad1c49b91fd4e9f880bb74bc7a65d8c",
  "status": "accepted",
  "status_url": "/api/v1/jobs/fad1c49b91fd4e9f880bb74bc7a65d8c",
  "document_url": "/api/v1/documents/fa0526d673424b3490680750290545f0"
}
```

### 4.2 Состояние задачи индексации

```bash
curl http://127.0.0.1:8000/api/v1/jobs/fad1c49b91fd4e9f880bb74bc7a65d8c
```

```json
{
  "job_id": "fad1c49b91fd4e9f880bb74bc7a65d8c",
  "doc_id": "fa0526d673424b3490680750290545f0",
  "status": "succeeded",
  "stage": "publish",
  "attempt": 1,
  "progress": {"completed": 13, "total": 13, "unit": "chunks"},
  "created_at": "2026-11-11T07:53:27.289157+00:00",
  "updated_at": "2026-11-11T07:53:34.582612+00:00",
  "error": null
}
```

Индексация этого документа (конвертация DOCX, разбор, эмбеддинги, запись,
проверка, публикация) заняла около 7 секунд.

### 4.3 Список документов

```bash
curl 'http://127.0.0.1:8000/api/v1/documents?limit=3'
```

```json
{
  "items": [
    {
      "doc_id": "fa0526d673424b3490680750290545f0",
      "title": "Договор поставки № ДП-2026/117",
      "status": "succeeded",
      "ready": true,
      "job_id": "fad1c49b91fd4e9f880bb74bc7a65d8c",
      "created_at": "2026-11-11T07:53:27.289157+00:00"
    },
    {
      "doc_id": "bc352d7dcc8649a78fc136c69e403d32",
      "title": "Smoke договор",
      "status": "succeeded",
      "ready": true,
      "job_id": "259c290381884c79b48dd4c81077139d",
      "created_at": "2026-11-11T07:49:58.091360+00:00"
    },
    {
      "doc_id": "ca3005103f0e4c30afde3e5e46616c3f",
      "title": "DOCX договор",
      "status": "succeeded",
      "ready": true,
      "job_id": "ae3bf28fe1254c9ab01080ea5193af86",
      "created_at": "2026-11-11T07:49:54.983742+00:00"
    }
  ],
  "next_cursor": "3"
}
```

### 4.4 Вопрос с точным обозначением (номер пункта), с диагностикой

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/fa0526d673424b3490680750290545f0/questions \
  -H 'Content-Type: application/json' \
  -d '{"question": "Какая неустойка предусмотрена пунктом 4.2 договора?", "debug": true}'
```

```json
{
  "answer": "За нарушение сроков поставки Поставщик уплачивает Покупателю неустойку в размере 0,1% от стоимости не поставленного в срок товара за каждый день просрочки, но не более 10% от общей цены договора. [1]",
  "found": true,
  "citations": [
    {
      "id": 1,
      "text": "4.2 За нарушение сроков поставки Поставщик уплачивает Покупателю\nнеустойку в размере 0,1% от стоимости не поставленного в срок товара за\nкаждый день просрочки, но не более 10% от общей цены договора.",
      "page": 1,
      "page_to": 1,
      "section": "4.2 За нарушение сроков поставки Поставщик уплачивает Покупателю",
      "chunk_id": "8864673f-5d16-51ac-8dfc-c9466215e26a",
      "char_start": 924,
      "char_end": 1123
    }
  ],
  "diagnostics": {
    "dense_returned": 13,
    "sparse_returned": 10,
    "fused_returned": 13,
    "reranked_count": 13,
    "context_count": 8,
    "context_chunk_ids": [
      "8864673f-5d16-51ac-8dfc-c9466215e26a",
      "1f9afbfa-ecd9-5422-aeed-262f6dc519bc",
      "e2d42baf-d291-5395-b177-c26124f26f93",
      "580489fd-1ad3-5a3b-ad67-d9aa49e421ce",
      "61190939-9955-514f-91ea-db2d7eba0809",
      "ede2921c-44d4-54bc-ba47-60b11e9812cb",
      "67a8622f-77c5-5264-b6b4-1ce82350ffa8",
      "875a9e53-607b-5164-a821-edb0c7a74ba0"
    ],
    "cited_chunk_ids": ["8864673f-5d16-51ac-8dfc-c9466215e26a"],
    "rerank_scores": {
      "8864673f-5d16-51ac-8dfc-c9466215e26a": 0.9893552660942078,
      "1f9afbfa-ecd9-5422-aeed-262f6dc519bc": 0.08538921177387238,
      "e2d42baf-d291-5395-b177-c26124f26f93": 0.01648861914873123
    },
    "stage_ms": {"encode": 614, "fusion": 3, "rerank": 45534, "context": 1},
    "model_ids": {
      "embedding": "ai-sage/Giga-Embeddings-instruct",
      "reranker": "Qwen/Qwen3-Reranker-8B",
      "llm": "qwen3.6:27b-q4_K_M"
    },
    "token_usage": {"input_tokens": 2251, "output_tokens": 5712, "llm_calls": 3},
    "branch_counts_source": "diagnostic_replay",
    "diagnostic_warnings": []
  }
}
```

Как читать диагностику: плотная ветвь дала 13 фрагментов (весь документ), ветвь
BM25 — 10; объединённый список из 13 кандидатов оценён реранкером; после
переранжирования в контекст LLM прошли 8 (`CONTEXT_TOP_K`); в ответе использован
один — фрагмент с пунктом 4.2, получивший оценку 0.989 против 0.085 у
следующего. `llm_calls: 3`: первый вариант ответа модели сервер отклонил при
проверке цитат и запросил одно исправление, затем отдельный вызов проверил
утверждение. Время ответа — 334 с (переранжирование на CPU — 45,5 с, остальное —
три вызова LLM).

### 4.5 Вопрос с точным обозначением (сумма)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/fa0526d673424b3490680750290545f0/questions \
  -H 'Content-Type: application/json' \
  -d '{"question": "Какова общая цена договора?"}'
```

```json
{
  "answer": "Общая цена договора составляет 4 850 000 (четыре миллиона восемьсот пятьдесят тысяч) рублей, включая НДС. [1]",
  "found": true,
  "citations": [
    {
      "id": 1,
      "text": "2.1 Общая цена договора составляет 4 850 000 (четыре миллиона восемьсот\nпятьдесят тысяч) рублей, включая НДС.",
      "page": 1,
      "page_to": 1,
      "section": "2.1 Общая цена договора составляет 4 850 000 (четыре миллиона восемьсот",
      "chunk_id": "ede2921c-44d4-54bc-ba47-60b11e9812cb",
      "char_start": 436,
      "char_end": 545
    }
  ]
}
```

Время ответа — 211 с.

### 4.6 Вопрос с точным обозначением (наименование стороны)

Запрос: `{"question": "Кто является Покупателем по договору?", "debug": true}`.
Ответ (блок `diagnostics` опущен: `dense_returned` 13, `sparse_returned` 8,
`context_count` 8, `llm_calls` 2):

```json
{
  "answer": "Покупателем по договору является акционерное общество «Уралпромснаб». [1]",
  "found": true,
  "citations": [
    {
      "id": 1,
      "text": "Общество с ограниченной ответственностью «Северный терминал»\n(Поставщик) и акционерное общество «Уралпромснаб» (Покупатель)\nзаключили настоящий договор о нижеследующем.",
      "page": 1,
      "page_to": 1,
      "chunk_id": "1f9afbfa-ecd9-5422-aeed-262f6dc519bc",
      "char_start": 69,
      "char_end": 237
    }
  ]
}
```

Цитата взята из преамбулы до первого нумерованного заголовка, поэтому ключа
`section` нет. Время ответа — 195 с.

### 4.7 Вопрос, ответа на который в документе нет

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/fa0526d673424b3490680750290545f0/questions \
  -H 'Content-Type: application/json' \
  -d '{"question": "Каков размер уставного капитала Поставщика?"}'
```

HTTP 200:

```json
{
  "answer": "В документе не найдено достаточно сведений для ответа на этот вопрос.",
  "found": false,
  "citations": []
}
```

Время ответа — 115 с.

### 4.8 Попытка передать историю диалога

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/fa0526d673424b3490680750290545f0/questions \
  -H 'Content-Type: application/json' \
  -d '{"question": "q", "history": ["prior"]}'
```

HTTP 422:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "body.history: Extra inputs are not permitted",
    "retryable": false,
    "request_id": "9dfb0faed14347f5ac2d2ab7aee5ec25"
  }
}
```

### 4.9 Готовность

```bash
curl http://127.0.0.1:8000/health/ready
```

```json
{"status": "ready", "components": {"redis": "ok", "qdrant": "ok", "local_models": "ok"}, "model_ready": true}
```

---

## 5. Конфигурация

Все настройки приложения — переменные окружения, которые читает
`app/config/settings.py` (pydantic-settings; невалидные значения — понятная
ошибка при старте). Локальный запуск не требует ни одной из них.

### Как значения попадают в сервисы

Переменную можно задать в shell или в файле `.env` в корне репозитория —
одинаково для `./run` и для `docker compose up`:

- **Настройки приложения** (поиск, разбиение, лимиты, повторы, подключения):
  `compose.yaml` передаёт их в контейнеры `api`, `worker`, `dispatcher`.
  Незаданная переменная в контейнер не попадает, и действует значение по
  умолчанию из `app/config/settings.py`.
- **Выбор моделей и профиля** (`LLM_PROVIDER`, `LLM_MODEL`, `EMBEDDING_MODEL`,
  `RERANKER_MODEL`, `QUALITY_PROFILE`, `PROFILE_STRICT`, `DEVICE`, `OFFLINE`):
  их читает `./run`, подбирает закреплённые ревизии и записывает решение в
  `.runtime/resolved.env` (переменные `B1_*`), которые подставляются в
  `compose.yaml`. Флаг CLI имеет приоритет над переменной.
- **Без launcher** (`docker compose up`) модели задаются напрямую переменными
  `B1_LLM_PROVIDER`, `B1_LLM_MODEL`, `B1_EMBEDDING_MODEL` + `B1_EMBEDDING_REVISION`,
  `B1_RERANKER_MODEL` + `B1_RERANKER_REVISION` + `B1_RERANKER_ADAPTER`; без них
  действуют умолчания compose.

Проверить, что получил сервис: `docker compose -p b1-rag exec api env`.

```bash
# пример: другой способ объединения и размеры ветвей
FUSION_METHOD=dbsf DENSE_TOP_K=40 SPARSE_TOP_K=30 RERANK_INPUT_K=50 CONTEXT_TOP_K=3 ./run
```

### Гибридный поиск и реранкер (требования задания)

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `FUSION_METHOD` | `rrf` | Способ объединения ветвей средствами Qdrant: `rrf` или `dbsf` |
| `DENSE_TOP_K` | `60` | Количество кандидатов плотной ветви |
| `SPARSE_TOP_K` | `60` | Количество кандидатов ветви BM25 |
| `RERANKER_MODEL` | `Qwen/Qwen3-Reranker-8B` | Модель реранкера: `Qwen/Qwen3-Reranker-8B` или `BAAI/bge-reranker-v2-m3` (то же, что `--reranker-model`; ревизию и адаптер подбирает launcher) |
| `RERANK_INPUT_K` | `60` | Количество кандидатов на входе реранкера (размер объединённого списка) |
| `CONTEXT_TOP_K` | `8` | Количество фрагментов на выходе реранкера, передаваемых в контекст LLM (верхняя граница; дополнительно действует `CONTEXT_BUDGET_TOKENS`) |
| `RERANKER_BATCH_SIZE` | `2` | Размер батча оценки в сервисе `local-models` |

Проверки при старте: `CONTEXT_TOP_K ≤ RERANK_INPUT_K ≤ DENSE_TOP_K + SPARSE_TOP_K`.

### Модели и LLM

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `LLM_PROVIDER` | автовыбор | `ollama` \| `dmr` \| `openai` \| `anthropic` (то же, что `--provider`) |
| `LLM_MODEL` | по профилю (`qwen3.6:27b-q4_K_M`) | Модель LLM (то же, что `--model`) |
| `EMBEDDING_MODEL` | по профилю (Giga-Embeddings-instruct) | Модель эмбеддингов (то же, что `--embedding-model`); входит в fingerprint индекса |
| `EMBEDDING_BATCH_SIZE` | `4` | Размер батча кодирования в сервисе `local-models` |
| `QUALITY_PROFILE`, `PROFILE_STRICT`, `DEVICE`, `OFFLINE` | `quality`, `false`, `auto`, `false` | Профиль качества и режим запуска (то же, что `--quality`, `--strict-profile`, `--device`, `--offline`) |
| `OLLAMA_BASE_URL` | по runtime | Адрес Ollama; определяется launcher (compose: `B1_OLLAMA_BASE_URL`) |
| `DMR_BASE_URL` | `http://model-runner.docker.internal/engines/v1` | Адрес Docker Model Runner (compose: `B1_DMR_BASE_URL`) |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | — | Ключи облачных провайдеров; используются только при явном выборе провайдера |
| `LLM_READ_TIMEOUT_S` | по runtime: 300, для CPU 1200 | Таймаут чтения ответа LLM; явное значение имеет приоритет над выбранным launcher |
| `QA_DEADLINE_S` | по runtime: 900, для CPU 3600 | Общий дедлайн вопроса; явное значение имеет приоритет |
| `LOCAL_CONTEXT_TOKENS` | `16384` | Окно контекста, под которое считается бюджет |
| `LLM_MAX_OUTPUT_TOKENS`, `LLM_REASONING_BUDGET_TOKENS` | `2048`, `4096` | Резерв под ответ и рассуждение |
| `LLM_THINKING` | по модели: `off` для `qwen3.5:9b/4b` (balanced, low-memory), иначе `auto` | Только Ollama: режим рассуждения модели — `auto` (умолчание модели), `on`, `off`. Умолчание `off` для 9B/4B выбрано по dev-сплиту этапа 9 (с рассуждением — пустые JSON-ответы и 502). Контекст Ollama всегда задаётся явно равным `LOCAL_CONTEXT_TOKENS` |
| `LLM_TEMPERATURE` | `0` | Только Ollama: температура генерации и верификатора; `presence_penalty` всегда 0, `seed` фиксирован — дословное цитирование воспроизводимо |
| `EMBEDDING_TOKENIZER_PATH` | по выбранному энкодеру | `tokenizer.json` энкодера для точного подсчёта токенов при разбиении; задаёт launcher (compose: `B1_TOKENIZER_PATH`) |

### Подключения

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant |
| `QDRANT_COLLECTION` | `document_chunks` | Единственная коллекция |
| `REDIS_URL` | `redis://redis:6379/0` | Реестр задач и документов, outbox |
| `CELERY_BROKER_URL` | `redis://redis:6379/1` | Брокер Celery |
| `CELERY_RESULT_BACKEND` | `redis://redis:6379/2` | Backend результатов Celery |
| `LOCAL_MODELS_URL` | `http://local-models:8081` | Сервис эмбеддингов и реранкера |
| `LOCAL_MODELS_TIMEOUT_S` | `300` | Таймаут его вызовов из API |
| `DOCUMENTS_DIR` | `/data/documents` | Хранилище загруженных файлов (в контейнерах фиксировано, это точка монтирования volume) |

### Разбиение, лимиты, надёжность

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `CHUNK_TOKENS`, `CHUNK_OVERLAP_TOKENS` | `450`, `64` | Размер фрагмента и перекрытие в токенах энкодера (входят в fingerprint индекса) |
| `CONTEXT_BUDGET_TOKENS` | `6000` | Бюджет токенов на фрагменты в контексте LLM |
| `BM25_LANGUAGE`, `BM25_AVG_LEN` | `russian`, `256` | Параметры BM25 |
| `MAX_FILE_MB`, `MAX_PAGES` | `25`, `100` | Лимиты документа |
| `MAX_QUESTION_CHARS` | `4000` | Максимальная длина вопроса |
| `MAX_CONCURRENT_QA` | `1` | Одновременные вопросы (остальные получают 429) |
| `MAX_LLM_CALLS_PER_QUESTION` | `4` | Общий бюджет вызовов LLM на вопрос: генерация + проверка + исправление + транспортные повторы |
| `MAX_REPAIR_ATTEMPTS` | `1` | Попытки исправления ответа модели |
| `LLM_MAX_ATTEMPTS` | `2` | Транспортные попытки одного вызова LLM |
| `QDRANT_MAX_ATTEMPTS` | `3` | Попытки одной операции Qdrant при индексации и при поиске |
| `JOB_MAX_ATTEMPTS` | `3` | Попытки задачи индексации |
| `RECONCILE_INTERVAL_S` | `30` | Период прохода реконсилера |
| `FAILED_SOURCE_TTL_S` | `604800` | Срок хранения исходника неуспешной задачи |
| `ALLOW_PAID_EVAL`, `EVAL_MAX_COST_USD` | `false`, — | Платные оценочные прогоны выключены; включение требует лимита расходов |

Полный список с комментариями — `.env.example`. Тест
`tests/unit/test_config_passthrough.py` следит, чтобы каждая настройка
приложения передавалась в контейнеры, а `.env.example` содержал только
существующие ключи.

---

## 6. Архитектура

### Сервисы (`compose.yaml`)

| Сервис | Роль |
|---|---|
| `api` | FastAPI: загрузка, статусы, список, вопросы. Единственный опубликованный порт (`127.0.0.1:8000`) |
| `dispatcher` | Переносит намерения из Redis-outbox в очередь Celery; периодически запускает реконсилер |
| `worker` | Celery-воркер индексации (concurrency 1) |
| `local-models` | Единственный владелец тяжёлых весов поиска: эмбеддинги и реранкер, HTTP `/v1/embed`, `/v1/rerank`, `/v1/health` |
| `models-init` | Разовая подготовка закреплённых ревизий моделей в volume `hf_models` |
| `ollama`, `ollama-init` | Контейнерный LLM-runtime и разовая загрузка модели (отключаются в профилях host Ollama / DMR / облако) |
| `redis` | Брокер Celery, реестр задач и документов, outbox, ключи идемпотентности, lease |
| `qdrant` | Векторное хранилище, одна коллекция `document_chunks` |

Реляционной БД нет. Volumes: `redis_data`, `qdrant_data`, `documents_data`,
`hf_models`, `ollama_models`, `fastembed_cache`.

### Индексация

```
POST /documents ─► файл в documents_data + запись задачи и outbox-намерение в Redis (атомарно) ─► 202
dispatcher ─► Celery ─► worker:
  DOCX → PDF (LibreOffice) → извлечение текста по страницам (pypdf) → канонический текст + карта страниц
  → разделы по нумерованным заголовкам → разбиение (LangChain RecursiveCharacterTextSplitter, длина в токенах энкодера)
  → dense-вектор (local-models) + sparse BM25 (fastembed) → upsert батчами в Qdrant под новым ingestion_run_id
  → проверка точного числа точек → run_complete на головном фрагменте → публикация (CAS под lease) ─► succeeded
```

Незавершённый запуск индексации невидим для вопросов: поиск фильтрует по
`doc_id` + опубликованному `ingestion_run_id` + fingerprint индекса. При сбое
точки незавершённого запуска удаляются (компенсация).

Владение задачей: воркер берёт lease в Redis до перехода задачи в `running` и
продлевает его фоновым heartbeat всё время выполнения, включая разбор документа.
Поэтому `running` без lease означает только упавший воркер: реконсилер либо
публикует уже проверенный запуск, либо переводит задачу в `retry_wait` и заново
отправляет её воркеру (не более `JOB_MAX_ATTEMPTS` попыток). Воркер, потерявший
lease, состояние задачи не меняет; повторная доставка задачи, у которой есть
владелец, ничего не делает.

### Payload фрагмента и индексы

Каждая точка Qdrant — один фрагмент с двумя векторами: именованный плотный
`dense` (Cosine) и разреженный `bm25` (IDF считает Qdrant).

| Поле | Описание |
|---|---|
| `doc_id` | Идентификатор документа |
| `doc_title` | Название документа из поля `title` загрузки |
| `chunk_index` | Порядковый номер фрагмента |
| `page_from`, `page_to` | Страницы фрагмента |
| `section` | Раздел (нумерованный заголовок), `null` до первого заголовка |
| `text` | Текст фрагмента |
| `char_start`, `char_end` | Смещения в каноническом тексте документа |
| `ingestion_run_id`, `index_fingerprint`, `is_document_head`, `run_complete` | Изоляция запусков индексации и совместимость индекса |
| `source_sha256`, `created_at`, `page_count`, `chunk_count`, `segments` | Сведения о документе и карта страниц фрагмента |

Payload-индексы: `doc_id`, `ingestion_run_id`, `index_fingerprint` (keyword),
`chunk_index` (integer), `is_document_head`, `run_complete` (bool) — все поля,
по которым выполняется фильтрация.

### Ответ на вопрос

```
POST /documents/{id}/questions ─► LangGraph (конечный граф, без checkpointer и thread_id):
  retrieve: эмбеддинг вопроса → ОДИН запрос Qdrant Query API: prefetch dense + prefetch bm25 → fusion (RRF/DBSF) на сервере
            → реранкер (обязателен) → верхние CONTEXT_TOP_K фрагментов в пределах бюджета токенов
  generate: LLM через LangChain (ChatOllama / ChatOpenAI / ChatAnthropic), структурированный вывод: утверждения + цитаты
  ground:   цитата — точная подстрока выданного фрагмента; числа утверждения присутствуют в цитатах
  verify:   второй вызов LLM — подтверждаются ли утверждения цитатами
  assemble: сервер собирает answer с маркерами [n] и citations со страницами и смещениями
```

Не более одной попытки исправления; все вызовы LLM, включая транспортные
повторы, укладываются в `MAX_LLM_CALLS_PER_QUESTION`.

### Обработка ошибок и повторы

- Единый слой `app/reliability/`: ограниченные повторы с экспоненциальной
  задержкой и jitter, учёт `Retry-After`, circuit breaker на зависимость,
  монотонные дедлайны по стадиям. Повторы SDK отключены, чтобы не умножаться.
- LLM: транспортные повторы (`LLM_MAX_ATTEMPTS`) в пределах общего бюджета
  вызовов; при недоступности — 503.
- Эмбеддинги и реранкер: повторы клиента `local-models`; недоступность реранкера
  — 503 без деградации до поиска без переранжирования.
- Qdrant: каждая операция — создание коллекции, запись батча, подсчёт и
  пометка при индексации, гибридный запрос при вопросе — повторяется до
  `QDRANT_MAX_ATTEMPTS` раз при сетевых сбоях и ответах 429/502/503/504; ошибки
  запроса (4xx) не повторяются. После исчерпания попыток: при вопросе — 503 с
  `retryable=true`, при индексации — откат незавершённого запуска и статус
  задачи `failed` с кодом `QDRANT_UNAVAILABLE`.
- Индексация: попытки считаются в Redis (`JOB_MAX_ATTEMPTS`), упавший воркер не
  оставляет частично видимый индекс.

---

## 7. Принятые решения

| Решение | Причина | Подробности |
|---|---|---|
| Одна коллекция, документы разделяются фильтром `doc_id` | Требование задания; payload-индексы на всех фильтруемых полях | `app/storage/qdrant_repo.py` |
| Dense + BM25 в одной точке, объединение одним запросом Qdrant (RRF по умолчанию, DBSF настройкой) | Точные обозначения ловит BM25, переформулировки — плотный вектор; склейки списков в коде нет | `app/retrieval/search.py` |
| Лексическая нормализация для BM25 (`ё→е`, синтетические токены для номеров пунктов, процентов, сумм) | Номер пункта «4.2» и «0,1%» должны находиться точным совпадением | `app/retrieval/lexical.py` |
| Обязательный реранкер, его отказ — 503 | Тихий откат к поиску без переранжирования дал бы незаметную потерю качества | `app/retrieval/search.py` |
| Счётчики ветвей в диагностике — отдельным повтором запроса | Fusion-запрос не возвращает размеры ветвей; лимиты за счётчики не выдаются | ADR 004 |
| Ответ собирает сервер из проверенных утверждений; цитата — точная подстрока документа | «Ответы только по документу»: модель не может выдать цитату, которой нет в тексте | `app/qa/grounding.py`, `app/qa/assemble.py` |
| Второй вызов модели проверяет поддержку утверждений | Дословная цитата ещё не гарантирует, что она подтверждает утверждение | `app/qa/graph.py` |
| Различение исходов: нет сведений → 200 `found=false`; сбой → 502/503/504 | Клиент должен отличать «в документе этого нет» от «сервис не смог ответить» | `app/api/questions.py` |
| Конечный LangGraph без checkpointer, сессий и `thread_id`; лишние поля запроса отклоняются | Требование «без истории» | `app/qa/graph.py`, `app/schemas/qa.py` |
| Индексация как saga: outbox в Redis, ключи идемпотентности, изоляция запусков, публикация CAS под lease, компенсация, реконсилер | Частично записанный индекс никогда не виден; устаревший воркер не может опубликовать или удалить принятый запуск | ADR 003 |
| Fingerprint индекса (энкодер и ревизия, разбиение, параметры BM25) | Индекс не пере-эмбеддится молча и не смешивает несовместимые векторы | `app/domain/index_fingerprint.py` |
| Один процесс `local-models` владеет весами поиска; API и воркер — клиенты | Одна копия тяжёлых моделей в памяти, отдельное окружение зависимостей | ADR 001, 002 |
| Local-first: ключ не включает облако, скрытого перехода local → cloud нет; поиск и реранкер локальны во всех режимах | Данные документов не уходят наружу без явного решения | ADR 005 |
| Закрепление версий: образы по digest, модели по ревизии, `uv.lock`, без `latest` | Воспроизводимость | ADR 001, `images.lock.json` |
| Номера страниц — по серверному PDF (DOCX конвертируется LibreOffice) | У DOCX нет собственных страниц; цитата должна указывать проверяемое место | `GET …/normalized` |

---

## 8. Доработки для production

1. **Доступ**: аутентификация, авторизация на уровне документа, tenant-scope во
   всех запросах, фильтрах и выдаче файлов. Сейчас `doc_id` — изоляция поиска, а
   не контроль доступа.
2. **Периметр**: TLS, управление секретами, квоты и rate limit на клиента,
   песочница для парсеров и LibreOffice, сканирование зависимостей, аудит без
   сырых текстов.
3. **Хранение**: объектное хранилище вместо локального volume; резервирование
   Redis и Qdrant, резервные копии и восстановление; восстановление реестра
   документов из payload Qdrant при потере Redis.
4. **Жизненный цикл индекса**: управляемая переиндексация при смене энкодера,
   удаление документов, сроки хранения.
5. **Масштабирование**: несколько реплик API и воркеров, общий (а не
   процессный) circuit breaker и лимит одновременных вопросов, выгрузка весов
   между стадиями, асинхронный режим вопросов для медленных моделей,
   `Retry-After` в ответах 429/503.
6. **Наблюдаемость**: метрики очереди, длительности стадий, доли отказов и
   невалидных ответов; трассировка; алерты.
7. **Качество**: ревью gold-набора человеком и слепой holdout, оценка профилей
   `quality` (27B, Qwen3-Reranker-8B в fp16/bf16), OCR для сканов, таблицы и сложная вёрстка.
8. **Эксплуатация**: проверка LLM-runtime и воркера в readiness, CI для
   launcher, API-ключ и настраиваемые таймауты подключения к внешнему Qdrant.

---

## 9. Что проверено

Ниже — сводка фактических прогонов.
Моковые тесты резолвера проверкой оборудования не считаются.

| Платформа / сценарий | Статус |
|---|---|
| macOS Apple Silicon (M4 Max), host Ollama (Metal) + контейнерный стек | проверено живым прогоном |
| macOS, контейнерный CPU-путь с чистых volumes, одной командой `./run` | проверено живым прогоном |
| Повторный старт без повторных загрузок | проверено |
| `./run --offline` (блокировка исходящего трафика, индексация внутри сети) | проверено |
| Docker Model Runner (`--provider dmr`) | QA проверен живым прогоном на этапе 6 |
| Linux CPU | логика резолвера покрыта unit-матрицей; живой запуск `not_run` (нет машины) |
| NVIDIA (`compose.gpu.yaml`) | `not_run` (нет GPU) |
| Windows (`run.ps1` → WSL2) | `not_run` (нет машины) |
| OpenAI / Anthropic | `not_run` (нет ключей; контракты запросов проверены без сети) |

### Оценка качества (этап 9)

Полный отчёт: [`eval/reports/REPORT.md`](eval/reports/REPORT.md) (сводка в начале),
воспроизведение — [`eval/README.md`](eval/README.md). Профиль balanced, отдельный
compose-проект `b1-eval`; судья — локальная qwen3.6:27b, платных вызовов нет.
Голден-сет: 133 вопроса по 14 документам из `test_documents` (93 с ответом, 40 без
ответа) + 17 проб изоляции; dev — документы 01/05/13/15, holdout — остальные 10.
Вопросы составлены агентом, цитаты проверены машинно, человеком не ревьюились.

| Итоговый пресет (thinking off, сопоставление цитат v2, temperature 0) | holdout | все |
|---|---|---|
| Найден ответ на вопрос с ответом | 60/69 | 79/93 |
| Верный отказ (`found=false`) на вопрос без ответа | 28/29 | 39/40 |
| Пробы изоляции (вопрос к чужому документу → `found=false`) | 14/14 | 17/17 |
| Ошибки 5xx | 0 | 0 |
| Валидность цитат (точная подстрока, сервер восстановил смещения) | 100% | 100% |
| Утечка цитат из другого документа | 0 | 0 |
| Полнота фактов по судье (все вопросы с ответом) | 71% | 69% |
| Ответы с запрещённым фактом | 0/69 | 0/93 |

Поиск (все 93 вопроса): гибрид RRF + реранкер — recall@8 0.968, hit@60 1.0;
dense 0.964, гибрид без реранкера 0.943, BM25 0.863. RAGAS итогового пресета
(holdout, 69 вопросов): faithfulness 0.970, factual correctness F1 0.660,
context recall 0.971, context precision 0.812.
Задержка ответа на стеке без посторонней нагрузки: p50 19 с, p95 45 с (9B), p50 15 с (4B).

Важные оговорки: оба исправления (сопоставление цитат и temperature 0) выбраны на dev,
но промахи holdout просматривались при диагностике — цифры holdout для них post-hoc,
а не слепая оценка. Qwen3-Reranker-8B — `not_run` (fp32 на MPS занял ~100 ГБ, остановлен).
Приёмка со свежей копии репозитория: [`eval/reports/acceptance/ACCEPTANCE.md`](eval/reports/acceptance/ACCEPTANCE.md).

**Состав gold-набора.** Категории: точное обозначение (номер пункта, сумма, сторона) — 30,
перефразированный вопрос — 29, несколько условий — 16, отрицание и ограничения — 18,
нет ответа в документе — 40. У каждого вопроса есть эталонный ответ, ожидаемые и
запрещённые факты и gold-фрагменты — точные цитаты с восстановленными смещениями и
страницами. Документ 06 — скан без текстового слоя, он проверяет только отказ
индексации с `OCR_REQUIRED`.

**Что измеряется.**

| Группа | Метрики | Как считается |
|---|---|---|
| Контракт и цитаты | валидность схемы, точная подстрока у каждой цитаты, утечка из другого документа | детерминированно по ответам API (`eval/metrics.py`) |
| Поиск | hit@60, recall@k, all-spans@8, nDCG@10, MRR; доля gold-фрагментов в контексте LLM | по gold-фрагментам и `context_chunk_ids` из `debug=true` |
| Исход ответа | найден / промах / верный отказ / 5xx; пробы изоляции | детерминированно |
| Содержание | полнота ожидаемых фактов, полные ответы, запрещённые факты, разбор каждого ложного found=true | локальный судья (`eval/judge.py`) |
| RAGAS | faithfulness, context recall, context precision, factual correctness | ragas 0.4.3, тот же локальный судья |
| Абляция и модели | dense / BM25 / гибрид RRF / DBSF / гибрид + реранкер; энкодеры и реранкеры на замороженных кандидатах | `eval/ablation.py`, `eval/compare_models.py` |

Варианты сравниваются парным sign-test по одним и тем же вопросам.

**Как воспроизвести** (кратко, полностью — в `eval/README.md`):

```bash
B1_PROJECT=b1-eval ./run --quality balanced          # отдельный стек, свои volumes
uv run python eval/upload_corpus.py                   # корпус + время индексации
python3 eval/export_index.py --project b1-eval && python3 eval/build_gold.py
uv run python eval/run_qa.py --run my-run             # вопросы через публичный API
uv run python eval/metrics.py --run my-run --split holdout
(cd eval && uv sync && .venv/bin/python judge.py --run my-run --split holdout)
python3 eval/report.py --runs my-run                  # eval/reports/REPORT.md
sh eval/demo.sh                                       # демо API с сохранением ответов
```

Артефакты: `eval/reports/REPORT.md` (сводка + таблицы по прогонам),
`eval/reports/runs/<прогон>/` (сырые ответы, метрики, вердикты судьи),
`eval/reports/ablation`, `eval/reports/compare`, `eval/reports/diagnosis`,
`eval/reports/demo`. Промежуточные данные (`eval/work/`) в git не входят.

Тесты: unit-набор без внешних сервисов; интеграционные — на реальных Redis,
Qdrant и сервисе моделей, включая инъекцию сбоев (`tests/integration/test_recovery_faults.py`:
падение воркера посреди индексации, сбой перед публикацией, устаревший токен,
недоступность Qdrant при компенсации, повторная доставка задачи).

---

## 10. Соответствие заданию и известные ограничения

| Требование задания | Статус | Где |
|---|---|---|
| Четыре операции API + диагностический параметр | выполнено | [раздел 3](#3-api-маршруты-и-json-контракты) |
| Ответ `answer` / `found` / `citations[text, page, section]`; `found=false` с пустым списком | выполнено | 3.6 |
| Ответы только по документу, без истории | выполнено | 3.6, раздел 6 |
| FastAPI, Celery + Redis, состояние задачи через API | выполнено | раздел 6 |
| Одна коллекция Qdrant, фильтр по `doc_id`, без реляционной БД | выполнено | раздел 6 |
| Payload-поля и payload-индексы | выполнено | раздел 6 |
| Два вектора в точке, объединение средствами Qdrant одним запросом | выполнено | раздел 6 |
| Обязательный реранкер, в контекст — только верхние фрагменты | выполнено | раздел 6 |
| Способ объединения, количество кандидатов ветвей, модель реранкера, его вход и выход — переменными окружения | выполнено | раздел 5 |
| Валидация ответа Pydantic | выполнено | `app/schemas/qa.py`, `app/qa/schemas.py` |
| Настройки моделей и подключений в переменных окружения | выполнено | раздел 5 |
| Обработка ошибок и повторы при недоступности LLM и Qdrant | выполнено | раздел 6 |

Известные ограничения текущей версии:

- **Эмбеддинги вызываются собственным HTTP-клиентом** сервиса `local-models`
  (`app/inference/client.py`), а не через интерфейс LangChain `Embeddings`.
- **Список документов строится по реестру в Redis.** Сведения о документе
  продублированы в payload, но при потере Redis проиндексированные документы
  перестают быть видны через API.
- **`doc_title` равен `null`**, если документ загружен без поля `title`.
- **Ключ `section` отсутствует в цитате** (вместо `null`), когда фрагмент
  находится до первого нумерованного заголовка.
- **`section` для нумерованных пунктов.** Абзац, начинающийся с номера пункта
  («4.2 За нарушение…»), распознаётся как заголовок, и в `section` попадает его
  первая строка, а не название раздела («4. Ответственность сторон») — см.
  примеры 4.4 и 4.5.
- **Потолок времени индексации** (15 минут на задачу) задан в коде.
- **Подключение к внешнему Qdrant** настраивается только адресом
  (`QDRANT_URL`): API-ключа и настраиваемых таймаутов нет.
- **Launcher:** `--provider` и `--device` не валидируются; `--residency`,
  `--allow-slow`, `--build` не влияют на поведение; `--quality`, `--model`,
  `--reranker-model`, `--port` не сохраняются между запусками; `--dry-run`
  записывает решение в `.runtime/`; готовность не проверяет LLM-runtime; путь к
  репозиторию с пробелом ломает профили с дополнительным compose-файлом; нет
  обнаружения устаревшей блокировки.
- Readiness и ответы 429/503 не содержат `Retry-After`; OpenAPI не описывает
  ответы с ошибками.

---

## 11. Разработка и тесты

Нужен `uv` (версии закреплены в `pyproject.toml` / `uv.lock`).

```bash
uv run pytest tests/unit -q                           # unit-тесты, без внешних сервисов
RUN_STACK_TESTS=1 uv run pytest tests/integration -q  # против запущенного стека
uv run ruff check app tests && uv run mypy            # линтер и типы
```

# TarotRAG — Intent Classifier

Baseline ML-компонент майбутнього таро-застосунку: визначає тему питання
(кохання / кар'єра / здоров'я / фінанси / духовність / загальне), щоб
пізніше звужувати RAG-пошук. Повний MLOps-цикл: tracking, model registry,
champion/challenger serving, автоматична оцінка нової моделі,
горизонтальне масштабування, оркестрований Airflow-пайплайн.

## Що є в стеку

| Сервіс | Навіщо | Порт |
|---|---|---|
| `api` (FastAPI) | `/predict`, `/feedback`, `/health` — сам класифікатор | через `nginx` |
| `nginx` | єдина точка входу, балансує трафік між репліками `api` | `8000` |
| `postgres` | таблиця `training_examples` (нові дані) + база `mlflow` (метадані MLflow) | `5432` |
| `minio` | S3-сумісне сховище: версії датасету + артефакти моделей MLflow | `9000` (API), `9001` (консоль) |
| `mlflow` | tracking server + model registry (аліаси `champion`/`challenger`) | `5000` |
| `airflow` | оркеструє DAG-и `sync_training_data` і `ml_pipeline` | `8080` |

## 1. Встановлення (один раз)

Потрібен лише **Docker Desktop** — усе інше (Python, бібліотеки, бази
даних) вже описано в `Dockerfile`'ах і піднімається автоматично.

1. Встанови Docker Desktop: <https://www.docker.com/products/docker-desktop>
2. Відкрий його, дочекайся зеленого "Running".
3. Перевір:
   ```bash
   docker --version && docker compose version
   ```

## 2. Налаштування

```bash
cd tarotrag
cp .env.example .env
```

Дефолтні значення в `.env` підходять для локальної роботи (не production-паролі).

## 3. Запуск

```bash
docker compose up -d --build
```

Автоматично відбудеться (нічого руками робити не треба):
1. Підіймуться `postgres` і `minio`.
2. `minio-init` створить бакети (`tarot-datasets`, `mlflow-artifacts`, `tarot-models`).
3. `dataset-init` опублікує `data/seed_intent_dataset.csv` як версію `v0-seed`.
4. `mlflow` підійметься (backend — Postgres, артефакти — MinIO).
5. `api` побачить, що моделі-champion ще нема, сама натренує baseline і зареєструє її.
6. `nginx` і `airflow` піднімуться окремо.

Перевірити стан:
```bash
docker compose ps
curl http://localhost:8000/health
```

### Якщо Postgres вже існував раніше (нова база `mlflow` не створилась)

Init-скрипти виконуються лише на **порожньому** томі даних. Якщо `postgres`
вже піднімався без бази `mlflow` — створи її вручну один раз:
```bash
docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d postgres -c "CREATE DATABASE mlflow;"'
```

## 4. Демо — задати питання системі

Через Swagger UI (найзручніше): <http://localhost:8000/docs> → `POST /predict` → Try it out.

Або з терміналу:
```bash
./scripts/demo.sh
```
чи вручну:
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"question": "Чи варто мені змінювати роботу цього року?"}' | python3 -m json.tool
```

Відповідь містить `intent`, `confidence`, `probabilities` по всіх 6 категоріях,
`served_by` (`champion` або `challenger`) і `model_version`.

## 5. Airflow — веб-інтерфейс

<http://localhost:8080>, логін `admin`, пароль — з логів контейнера:
```bash
docker compose logs airflow | grep -i password
```

Два DAG-и (обидва потрібно увімкнути тумблером перед першим запуском):
- **`sync_training_data`** — переносить нові рядки з Postgres (`/feedback`, симуляція) у нову версію датасету в MinIO.
- **`ml_pipeline`** — повний конвеєр: `gather_data` → `process_data` → `tune_and_train` (grid search) → `evaluate_and_promote` (авто-промоушн, якщо нова модель краща за champion).

Запуск з терміналу:
```bash
docker compose exec airflow airflow dags trigger ml_pipeline
docker compose exec airflow airflow dags list          # перевірити точні назви DAG-ів
```

## 6. MLflow UI — метрики й версії моделі

<http://localhost:5000> — список прогонів, параметри, метрики (`accuracy`,
`macro_f1`), Model Registry з версіями `tarot-intent-classifier` та
аліасами `champion`/`challenger`.

## 7. Цикл покращення моделі вручну (без Airflow)

```bash
docker compose run --rm simulate-new-labels   # вдає нові розмічені дані
# ...або тригерни sync_training_data / ml_pipeline в Airflow UI...
docker compose run --rm train                 # тренує + тюнить + реєструє як challenger
docker compose run --rm promote-challenger    # ручний промоушн (override оцінки)
```
API сама підхопить зміну за `MODEL_REFRESH_SECONDS` (типово — 60 секунд), або
негайно: `curl -X POST http://localhost:8000/admin/reload-models`.

## 8. Масштабування

```bash
docker compose up -d --scale api=3
```
`nginx` сам розподілить трафік між усіма живими репліками `api`.
Повернутись до однієї: `docker compose up -d --scale api=1`.

## 9. Дивитись, що реально лежить у сховищах

```bash
# Postgres
docker compose exec postgres psql -U tarotrag -d tarotrag -c "SELECT * FROM training_examples ORDER BY id DESC LIMIT 10;"

# MinIO (браузер, простіше): http://localhost:9001  (MINIO_ROOT_USER / MINIO_ROOT_PASSWORD з .env)
```

## Структура проєкту

```
tarotrag/
├── docker-compose.yml          диригент: усі сервіси й залежності
├── .env / .env.example         конфігурація
├── data/                       seed-датасет (120 прикладів)
├── common/                     спільний код: Postgres (db.py), MinIO (storage.py)
├── training/train_baseline.py  тренування, тюнінг, реєстрація в MLflow
├── services/api/               FastAPI-сервер (Intent Classifier)
├── services/mlflow/            власний MLflow-сервер
├── airflow/                    Airflow-образ + два DAG-и
├── infra/nginx/                балансувальник навантаження
└── scripts/                    одноразові допоміжні скрипти
```

## Відомі обмеження (свідомі спрощення для курсового проєкту)

- Airflow працює на `SequentialExecutor` (один контейнер, одне завдання
  одночасно) — тому кроки, що чекають на дочірній DAG
  (`TriggerDagRunOperator`), обов'язково мають `deferrable=True`, інакше
  виникає deadlock.
- Паролі й ключі в `.env` — dev-only, не для продакшну.
- Seed-датасет (120 прикладів) — навчальний, тому baseline-метрики
  (macro-F1 ≈ 0.45–0.68 залежно від прогону) скромні; це очікувано.

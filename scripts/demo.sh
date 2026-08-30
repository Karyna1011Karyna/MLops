#!/usr/bin/env bash
# Demo the running API
# docker compose up -d
set -euo pipefail

API="${API_URL:-http://localhost:8000}"

echo "== health =="
curl -s "$API/health" | python3 -m json.tool

echo
echo "== predict: kohannia (кохання) =="
curl -s -X POST "$API/predict" -H "Content-Type: application/json" \
  -d '{"question": "Чи варто мені писати першою після сварки з партнером?"}' | python3 -m json.tool

echo
echo "== predict: kariera (кар'єра) =="
curl -s -X POST "$API/predict" -H "Content-Type: application/json" \
  -d '{"question": "Чи варто мені змінювати роботу цього року?"}' | python3 -m json.tool

echo
echo "== predict: finansy (фінанси) =="
curl -s -X POST "$API/predict" -H "Content-Type: application/json" \
  -d '{"question": "Чи вдасться мені розрахуватися з боргами цього року?"}' | python3 -m json.tool

echo
echo "== feedback: додаємо нове розмічене питання для Airflow =="
curl -s -X POST "$API/feedback" -H "Content-Type: application/json" \
  -d '{"question": "Чи варто мені братися за цей новий проєкт на роботі?", "label": "kariera"}' | python3 -m json.tool

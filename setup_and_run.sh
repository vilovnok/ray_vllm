#!/bin/bash
set -e

# Перейти в директорию проекта
cd serve || exit

# Установить зависимости
echo "Installing libraries from requirements.txt..."
pip install --no-cache-dir -r requirements.txt

# Перейти в исходники
cd src || exit

# Запустить деплоймент
echo "Running deployment 'my_app'..."
serve run vllm_serve:llm_app --name my_app

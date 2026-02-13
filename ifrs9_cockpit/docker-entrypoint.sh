#!/bin/bash
set -e

case "$1" in
    dashboard)
        echo "Starting IFRS 9 Risk Cockpit Dashboard..."
        exec streamlit run ifrs9_cockpit/app.py \
            --server.port=${STREAMLIT_SERVER_PORT:-8501} \
            --server.address=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0} \
            --server.headless=true
        ;;
    api)
        echo "Starting IFRS 9 Risk Cockpit API..."
        exec uvicorn ifrs9_cockpit.api:app \
            --host 0.0.0.0 \
            --port ${API_PORT:-8000} \
            --workers ${API_WORKERS:-2}
        ;;
    both)
        echo "Starting Dashboard + API..."
        uvicorn ifrs9_cockpit.api:app \
            --host 0.0.0.0 \
            --port ${API_PORT:-8000} \
            --workers ${API_WORKERS:-2} &
        exec streamlit run ifrs9_cockpit/app.py \
            --server.port=${STREAMLIT_SERVER_PORT:-8501} \
            --server.address=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0} \
            --server.headless=true
        ;;
    *)
        exec "$@"
        ;;
esac

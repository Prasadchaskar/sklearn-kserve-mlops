FROM seldonio/mlserver:1.5.0

WORKDIR /app

RUN pip install --no-cache-dir mlflow==2.10.2

COPY src/serving/mlflow_storage_initializer.py \
     /app/mlflow_storage_initializer.py

ENTRYPOINT ["python", "/app/mlflow_storage_initializer.py"]

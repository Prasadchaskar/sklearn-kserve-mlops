FROM seldonio/mlserver:1.5.0

WORKDIR /app

RUN pip install --no-cache-dir \
    mlflow==3.16.1 \
    numpy==2.5.3 \
    pandas==3.0.6 \
    scikit-learn==1.9.1 \
    scipy==1.18.1 \
    skops==0.16.0

EXPOSE 8080 9000

CMD ["mlserver", "start", "/mnt/models"]
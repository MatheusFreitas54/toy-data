FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas scikit-learn tensorflow minio
COPY ./movies /app/movies

CMD ["python", "/movies/modelo_preditivo.py"]
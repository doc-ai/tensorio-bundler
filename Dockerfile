FROM python:3.6.8-slim

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "--bind=0.0.0.0:8000", "--log-file=-", "tensorio_bundler.rest:api"]

FROM python:3.6-slim

COPY . .

WORKDIR /root

RUN pip install flask gunicorn keras numpy tensorflow sklearn joblib flask_wtf
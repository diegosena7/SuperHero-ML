FROM python:3.10-slim

WORKDIR /app

COPY ./codigo /app
COPY dados/ ./dados

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
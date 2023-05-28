FROM python:3.11.3-slim-buster

WORKDIR /app

COPY app.py .
COPY requirements.txt .
COPY model.pkl .
COPY modelapp.py .

RUN pip install -r requirements.txt

EXPOSE 5050

ENTRYPOINT ["python"]

CMD ["app.py"]

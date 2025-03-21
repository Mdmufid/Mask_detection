FROM python:3.9

WORKDIR /app
COPY . /app

RUN pip install fastapi uvicorn tensorflow opencv-python numpy
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 👇 COPY FRONTEND FILES (THIS WAS MISSING)
COPY app.py .
COPY templates ./templates
COPY static ./static

EXPOSE 7860

CMD ["python", "app.py"]

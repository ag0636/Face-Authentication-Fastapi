# ✅ Uses dlib preinstalled (officially tested)
FROM akhilnarang/dlib-opencv:python3.10

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

# Skip installing dlib and opencv again (already included)
# So remove them from requirements.txt if needed
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

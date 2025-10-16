FROM python:3.10-slim
RUN pip install --no-cache-dir --upgrade pip
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY train.py /app/train.py
VOLUME ["/data", "/artifacts"]
ENTRYPOINT ["python", "train.py"]

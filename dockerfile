FROM python:3.10-slim

WORKDIR /tz_sber

COPY requirements.txt .

# ХИТРОСТЬ: Ставим облегченную CPU-версию PyTorch прямо из официального репозитория
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
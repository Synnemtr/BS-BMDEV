FROM python:3.8.14

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

# COPY . .
# The main file to be run should be replaced with the "identify_attacker.py"
CMD ["python", "identify_attacker.py"]
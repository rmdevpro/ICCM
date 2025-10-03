FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy KGB application code
COPY kgb/ ./kgb/

# Expose KGB port
EXPOSE 9000

# Run KGB spy coordinator
CMD ["python", "-m", "kgb.proxy_server"]

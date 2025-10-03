FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy KGB application code
COPY kgb/ ./kgb/

# Expose KGB ports (WebSocket spy + HTTP gateway)
EXPOSE 9000 8089

# Run KGB (WebSocket spy + HTTP gateway)
CMD ["python", "-m", "kgb.proxy_server"]

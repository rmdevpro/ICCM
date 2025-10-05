# Dockerfile
FROM python:3.11-alpine

# Install system dependencies for python-magic and fcntl
RUN apk add --no-cache libmagic build-base

WORKDIR /app

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY test_iccm_network.py .

# Expose the MCP port
EXPOSE 8070

# Run test server (tests iccm-network library without database)
# Copy iccm-network to writable location and run test
CMD ["sh", "-c", "cp -r /app/iccm-network/iccm_network /app/ && python test_iccm_network.py"]

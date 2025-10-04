# Dockerfile for Claude Code UI
# Wraps claudecodeui to provide web interface for containerized Claude (Claudette)

FROM node:20-alpine

# Install Docker CLI and build dependencies for native modules
RUN apk add --no-cache \
    docker-cli \
    python3 \
    make \
    g++

# Set working directory
WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy application code
COPY . .

# Build the frontend
RUN npm run build

# Expose ports
# 3001 - Backend API server
# 5173 - Frontend (Vite dev server, not used in production)
EXPOSE 3001

# Start the application
CMD ["npm", "start"]

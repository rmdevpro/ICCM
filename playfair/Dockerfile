# Phase 1: Build Environment
# Use official Node.js 22 image on Debian Bookworm for build stability
FROM node:22-bookworm-slim as builder

# Set working directory
WORKDIR /app

# Copy package files and install dependencies
COPY package.json ./
# Use npm install (package-lock.json not yet generated)
RUN npm install --omit=dev

# Copy application code
COPY *.js ./playfair/
COPY utils/ ./playfair/utils/
COPY config/ ./playfair/config/
COPY themes/ ./playfair/themes/
COPY engines/ ./playfair/engines/
COPY examples/ ./playfair/examples/
COPY workers/ ./playfair/workers/

# Phase 2: Production Environment
# Use Ubuntu 24.04 as requested for the final image
FROM ubuntu:24.04

# Set non-interactive frontend to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for Playfair
# - graphviz: For DOT rendering
# - fonts-inter, fonts-roboto: For modern theming
# - curl: For health checks
# - ca-certificates: For secure connections if needed later
# - libcairo2: Required by graphviz for high-quality rendering
# - nodejs & npm: For running the application
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    fonts-inter \
    fonts-roboto \
    curl \
    ca-certificates \
    libcairo2 \
    nodejs \
    npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Mermaid CLI globally using the installed npm
# This ensures it's available in the PATH for the application
RUN npm install -g @mermaid-js/mermaid-cli@11

# Create a non-root user for security (before copying files)
RUN useradd --system --create-home --shell /bin/false appuser

# Create app directory and set ownership
WORKDIR /app
RUN chown -R appuser:appuser /app

# Switch to non-root user before copying files
USER appuser

# Copy built dependencies and application code from the builder stage with correct ownership
COPY --from=builder --chown=appuser:appuser /app/node_modules ./node_modules
COPY --from=builder --chown=appuser:appuser /app/playfair ./playfair
COPY --chown=appuser:appuser package.json .

# Expose the application port (container-side)
EXPOSE 8040

# Health check to ensure the service is running correctly
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8040/health || exit 1

# Command to run the application
CMD ["node", "playfair/server.js"]
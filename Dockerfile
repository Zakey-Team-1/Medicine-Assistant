# Build frontend assets using a Node builder stage (Tailwind)
FROM node:18-alpine AS node_builder
WORKDIR /build

# Copy package manifest and Tailwind config then install deps
COPY package.json package-lock.json* tailwind.config.js ./

# Copy source CSS and related files needed for build
COPY src ./src

RUN npm ci && npm run build

# Use official Python runtime as base image for the app
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/

# Copy built frontend assets from the Node builder
COPY --from=node_builder /build/src/web/static/css/output.css /app/src/web/static/css/output.css

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Expose port
EXPOSE 8002

# Set environment variables
# Point FLASK_APP at the module and app object so the Flask CLI can import it
ENV FLASK_APP=src.web.app:app
ENV FLASK_ENV=production

# Run the Flask application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=8002"]

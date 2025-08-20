# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Install uv
RUN pip install uv

# Copy the dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the API server
CMD ["uv", "run", "main.py", "start-api", "--host", "0.0.0.0", "--port", "8000"]

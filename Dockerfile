# Use a small Python base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the container
COPY . .

# Expose the port your Flask app uses
EXPOSE 9696

# Start the web service with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]


# Use a Python image as the base image (you can use python:3.11 or another version)
FROM python:3.11

# Install required system dependencies for building sentencepiece
RUN apt-get update && \
    apt-get install -y \
    cmake \
    pkg-config \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy your requirements.txt file to the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY modified_packages/pipeline_base.py /usr/local/lib/python3.11/site-packages/openprompt/

# Copy the rest of your application files
COPY . .

# Expose the port Django runs on
EXPOSE 8000

# Run migrations and start the server
CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]

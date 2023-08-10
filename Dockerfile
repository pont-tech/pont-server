# Use an official Python runtime as the parent image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose the port on which the Streamlit app will run (default is 8501)
EXPOSE 5000

# Set the entrypoint command to run the Streamlit app
CMD ["flask", "--app", "server.py" "run", "--host=0.0.0.0"]

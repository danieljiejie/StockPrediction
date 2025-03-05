# Use an official Python base image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy all project files to the container
COPY . .

# Install dependencies
RUN pip install streamlit yfinance plotly pandas scikit-learn numpy ta xgboost tensorflow textblob requests

# Expose port 5000 (matches your config.toml)
EXPOSE 5000

# Command to run your app
CMD ["streamlit", "run", "main.py", "--server.port=5000", "--server.address=0.0.0.0"]
CMD ["streamlit", "run", "main.py", "--server.port=5000", "--server.address=0.0.0.0"]
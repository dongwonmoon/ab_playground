# Use a Miniconda base image
FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Copy environment definition and project files
COPY . .

# Create the conda environment from the requirements.txt file
# Note: This can be slow. For faster builds, consider creating a more specific environment.yml
RUN conda create --name ab_playground --file requirements.txt python=3.9 -y

# Activate the conda environment
SHELL ["conda", "run", "-n", "ab_playground", "/bin/bash", "-c"]

# Expose ports for Streamlit and MLflow
EXPOSE 8501
EXPOSE 5000

# Default command to run when starting the container (can be overridden)
CMD ["echo", "Container is ready. Please use docker-compose to run services."]

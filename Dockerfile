# Use the official Gurobi Python Docker image
FROM gurobi/python:latest

# Set environment variables
ENV APP_HOME /app
ENV MODULE "grid_edge_optimizer"

# Create the project directory structure
RUN mkdir -p $APP_HOME/$MODULE

# Set the working directory inside the container
WORKDIR $APP_HOME/$MODULE

# Install dependencies in the global Python environment
COPY src/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies like numpy and scipy
RUN pip install numpy scipy

# Copy the Python source code and Gurobi license into the container
COPY src/ $APP_HOME/$MODULE/src/
COPY gurobi.lic /opt/gurobi/gurobi.lic

# Expose the port for the FastAPI application
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

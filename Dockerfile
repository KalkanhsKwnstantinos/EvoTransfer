FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Set user and workdir
USER 1000:1000
WORKDIR /workspace

# Install the application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose Jupyter port
EXPOSE 8888

CMD ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.token="]
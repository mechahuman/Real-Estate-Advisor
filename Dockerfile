# Use a stable Python 3.11 base (small image)
FROM python:3.11-slim

# Install system build deps needed by some scientific packages
# Keep the list minimal; this is enough to compile if needed.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      gfortran \
      libatlas-base-dev \
      && rm -rf /var/lib/apt/lists/*

# Ensure pip/setuptools/wheel are recent so wheels are preferred
RUN python -m pip install --upgrade pip setuptools wheel

# Create non-root user for better security
RUN useradd --create-home appuser
WORKDIR /home/appuser/app
USER appuser

# Copy only requirements first to leverage Docker layer cache
# (Make sure requirements.txt exists in repo root)
COPY --chown=appuser:appuser requirements.txt ./requirements.txt

# Install Python dependencies preferring wheels (faster & avoids building)
# You can change --prefer-binary as needed
RUN pip install --prefer-binary -r requirements.txt

# Copy rest of application code
COPY --chown=appuser:appuser . .

# Expose the Streamlit default port (optional; for local docker run)
EXPOSE 8501

# Recommended Streamlit run command: use $PORT if provided by host (e.g. cloud)
# This command allows Streamlit to accept the externally provided PORT env var,
# or fall back to 8501 for local testing.
CMD ["bash", "-lc", "streamlit run app.py --server.port=${PORT:-8501} --server.headless=true"]

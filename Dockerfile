FROM python:3.8

# Create the directories
RUN mkdir -p app/ app/tiles/

# Install the dependencies
RUN pip3 install git+https://github.com/cytomine-uliege/Cytomine-python-client.git@v2.8.3
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

# Copy the scripts
COPY descriptor.json /app/descriptor.json
COPY run.py /app/run.py

ENTRYPOINT ["python3", "/app/run.py"]
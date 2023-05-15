# Starting from an official AWS image
# Keep any dependencies and versions in this file aligned with the environment.yml and Makefile
FROM public.ecr.aws/lambda/python:3.10

# Install Python dependencies
COPY requirements/prod.txt ./requirements.txt
RUN pip install --upgrade pip==23.1.2
RUN pip install -r requirements.txt

# Copy only the relevant directories and files
#   note that we use a .dockerignore file to avoid copying logs etc.
COPY text_recognizer/ ./text_recognizer
COPY api_serverless/api.py ./api.py

CMD ["api.handler"]

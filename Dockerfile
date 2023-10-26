FROM python:3.11

Run pip install pipenv

workdir /app

copy main02.py ./

copy Pipfile Pipfile.lock ./

run pipenv install --system --deploy --ignore-pipfile

Expose 8000
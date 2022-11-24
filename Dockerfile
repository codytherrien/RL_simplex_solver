FROM python:3.9.3

COPY requirements.txt /tmp/

RUN pip install -r /tmp/requirements.txt

RUN useradd --create-home appuser
WORKDIR /home/appuser
USER appuser


COPY simplex_model.h5 .
COPY run_simplex_agent.py .
COPY simplex_q_network.py .
COPY Simplex.py .

ENTRYPOINT ["python", "run_simplex_agent.py"]
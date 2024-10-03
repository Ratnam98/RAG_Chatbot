FROM python:3.12.4

WORKDIR /app

COPY requirements.txt ./

COPY Openai_1.py ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt --default-timeout=100

COPY . .

EXPOSE 8501

CMD ["streamlit","run","Openai_1.py","--server.port=8501","--server.address=0.0.0.0"]

#ENTRYPOINT ["streamlit","run","Openai_1.py","--server.port=8501","--server.address=0.0.0.0"]
FROM python:2.7
ADD . /app
WORKDIR /app
EXPOSE 4000
RUN pip install -r requirements.txt
CMD python app/index.py
#ENTRYPOINT ["python","app/index.py"]



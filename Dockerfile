FROM python:2.7.11
COPY TEST.py .
RUN python TEST.py
CMD ["python","TEST"] 

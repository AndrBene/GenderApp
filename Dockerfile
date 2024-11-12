FROM python:3.11-slim
WORKDIR /usr/src/app
COPY . .
RUN echo "Debugging Dockerfile build"
CMD ["python", "-c", "print('Hello, Back4app')"]
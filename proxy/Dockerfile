FROM ghcr.io/berriai/litellm:main-latest

WORKDIR /app
COPY config.yaml .

EXPOSE 4000

CMD ["--config", "config.yaml", "--port", "4000", "--host", "0.0.0.0"]

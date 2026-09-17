# Disposable fixture only: a self-signed leaf certificate generated fresh at build time, never
# in any OS trust store. Backs crates/elm-connectors/tests/tls.rs's real certificate-verification
# tests. Never reuse this image or its key outside a throwaway test run.
FROM postgres:16
RUN apt-get update && apt-get install -y --no-install-recommends openssl && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /certs \
 && openssl req -x509 -newkey rsa:2048 -days 2 -nodes \
      -keyout /certs/server.key -out /certs/server.crt \
      -subj "/CN=localhost" \
      -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" \
 && cp /certs/server.crt /certs/ca.crt \
 && chown postgres:postgres /certs/server.key /certs/server.crt /certs/ca.crt \
 && chmod 600 /certs/server.key \
 && chmod 644 /certs/server.crt /certs/ca.crt
CMD ["postgres", "-c", "ssl=on", "-c", "ssl_cert_file=/certs/server.crt", "-c", "ssl_key_file=/certs/server.key"]

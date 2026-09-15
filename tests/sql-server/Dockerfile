FROM rust:1-bookworm

# Native clients stay in this development image, never in the product bundle.
RUN curl --fail --silent --show-error --location \
      https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb \
      --output /tmp/packages-microsoft-prod.deb \
    && dpkg -i /tmp/packages-microsoft-prod.deb \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install --yes --no-install-recommends msodbcsql18 unixodbc-dev \
    && rm /tmp/packages-microsoft-prod.deb

WORKDIR /workspace
ENV CARGO_TARGET_DIR=/build
CMD ["cargo", "test", "--locked", "-p", "elm-connectors", "--features", "sql-server", "--test", "sql_server", "--", "--ignored", "--test-threads=1"]

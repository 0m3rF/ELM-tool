FROM rust:1-bookworm

# Development-only native clients; never included in the product bundle.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libaio1 unzip unixodbc-dev \
    && curl --fail --silent --show-error --location --retry 3 --retry-all-errors \
      https://download.oracle.com/otn_software/linux/instantclient/2123000/instantclient-basic-linux.x64-21.23.0.0.0dbru.zip \
      --output /tmp/instantclient.zip \
    && echo '68a6ccd7ca6fbfb4d2914bd6531a8599fdb75841b8d47df5256bfef40d020820  /tmp/instantclient.zip' | sha256sum --check \
    && unzip -q /tmp/instantclient.zip -d /opt/oracle \
    && rm /tmp/instantclient.zip

RUN curl --fail --silent --show-error --location --retry 3 --retry-all-errors \
      https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb \
      --output /tmp/packages-microsoft-prod.deb \
    && dpkg -i /tmp/packages-microsoft-prod.deb \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install --yes --no-install-recommends msodbcsql18 \
    && rm /tmp/packages-microsoft-prod.deb

WORKDIR /workspace
ENV LD_LIBRARY_PATH=/opt/oracle/instantclient_21_23
ENV CARGO_TARGET_DIR=/build
CMD ["cargo", "test", "--locked", "-p", "elm-connectors", "--features", "all-databases", "--test", "cross_database", "--", "--ignored", "--test-threads=1"]

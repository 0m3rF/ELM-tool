FROM rust:1-bookworm

# Development-only native client; never included in ELM release bundles.
# Oracle's published Basic ZIP SHA-256 is checked before extraction.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libaio1 unzip unixodbc-dev libdbus-1-dev dbus-x11 gnome-keyring \
    && curl --fail --silent --show-error --location \
      https://download.oracle.com/otn_software/linux/instantclient/2123000/instantclient-basic-linux.x64-21.23.0.0.0dbru.zip \
      --output /tmp/instantclient.zip \
    && echo '68a6ccd7ca6fbfb4d2914bd6531a8599fdb75841b8d47df5256bfef40d020820  /tmp/instantclient.zip' | sha256sum --check \
    && unzip -q /tmp/instantclient.zip -d /opt/oracle \
    && rm /tmp/instantclient.zip

WORKDIR /workspace
ENV LD_LIBRARY_PATH=/opt/oracle/instantclient_21_23
ENV CARGO_TARGET_DIR=/build
CMD ["sh", "tests/oracle/run.sh"]

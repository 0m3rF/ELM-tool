FROM rust:1-bookworm

WORKDIR /workspace
ENV CARGO_TARGET_DIR=/build
CMD ["cargo", "test", "--locked", "-p", "elm-connectors", "--test", "full_disk", "--", "--ignored", "--test-threads=1"]

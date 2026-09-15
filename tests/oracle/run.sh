#!/bin/sh
set -eu
cargo test --locked -p elm-connectors --features oracle --test oracle -- --ignored --test-threads=1
dbus-run-session -- sh tests/oracle/run-daemon.sh

#!/bin/sh
set -eu
# Empty unlock password is confined to this disposable development container.
printf '\n' | gnome-keyring-daemon --unlock --components=secrets >/dev/null
cargo test --locked -p elm-daemon --test oracle -- --ignored --test-threads=1

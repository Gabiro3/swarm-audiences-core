#!/bin/sh
set -e

# Railway (and most volume mounts) attach the volume owned by root at
# container start, which shadows whatever the Dockerfile's build-time chown
# set up — that chown only affects what's baked into the image layer, not
# data mounted in afterwards. So fix ownership here, at runtime, every start,
# then drop from root to appuser before exec'ing the real command.
if [ "$(id -u)" = "0" ]; then
    chown -R appuser:appuser "$HOME" /app
    exec gosu appuser "$@"
fi

exec "$@"

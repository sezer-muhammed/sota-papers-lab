#!/usr/bin/env bash
# Baseline adapter: gsplat simple_trainer `mcmc` (3DGS-MCMC, 1M Gaussian cap by
# default; raise it with `-- --strategy.cap-max 3000000`). See gsplat.sh.
GSPLAT_CONFIG=mcmc exec "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/gsplat.sh" "$@"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

case "${1:-all}" in
    all)
        cargo test
        ;;
    kb)
        cargo test --test kb_page_index_cli
        cargo test kb_
        ;;
    test)
        shift
        cargo test "$@"
        ;;
    *)
        echo "usage: $0 [all|kb|test <cargo-test-args...>]" >&2
        exit 2
        ;;
esac

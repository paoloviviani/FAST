#!/bin/bash
FAST_ROOT=$(cd ../..; pwd)

$FAST_ROOT/bin/fast-submit -H hosts -l local -n 9 $@
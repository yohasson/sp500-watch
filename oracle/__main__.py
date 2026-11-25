import argparse, sys
from . import pipeline
p = argparse.ArgumentParser(prog="oracle")
p.add_argument("cmd", choices=["run"], help="run = update data, replay, seal, grade, build dashboard")
p.add_argument("--no-update", action="store_true"); p.add_argument("--no-seal", action="store_true")
p.add_argument("--offline-universe", action="store_true")
a = p.parse_args()
st = pipeline.run(update=not a.no_update, seal=not a.no_seal, offline_universe=a.offline_universe)
sys.exit(0)

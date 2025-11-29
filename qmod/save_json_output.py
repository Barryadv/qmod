from qmod.pipeline import run_once

out = run_once({'tkr':'AAPL','start':'2024-01-01','end':'2024-06-01'})
with open("run_output.json", "w", encoding="utf-8") as f:
    f.write(out.to_json(indent=2))
print("Wrote run_output.json")

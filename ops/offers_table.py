"""Print Vast.ai offers (JSON on standard input) as a table, cheapest per unit of work first. Used by vast_find.sh."""
import json
import sys

offers = sorted(json.load(sys.stdin), key=lambda x: x["dph_total"] / max(x.get("dlperf") or 1, 1))
print(f"{'offer':>10} {'card':9} {'$/hour':>7} {'$/work':>7} {'cores':>5} {'GB RAM':>6} {'up/down Mbit':>13} {'reliab.':>7}  place")
for x in offers[:12]:
    print(f"{x['id']:>10} {x['gpu_name']:9} {x['dph_total']:7.3f} {1000 * x['dph_total'] / x['dlperf']:7.2f} "
          f"{x['cpu_cores_effective']:5.0f} {x['cpu_ram'] / 1024:6.0f} {x['inet_up']:6.0f}/{x['inet_down']:<6.0f} "
          f"{x.get('reliability2', 0):7.4f}  {x.get('geolocation')}")
print("\n$/hour includes a 100 GB disk. $/work = price divided by Vast.ai's speed rating (lower is better).")
print("A machine in Europe uploads faster from this server (Germany): 48 MB/s to Estonia, 14 MB/s to Texas.")

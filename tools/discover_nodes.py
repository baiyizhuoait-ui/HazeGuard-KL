#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
节点发现：在马来半岛 + 苏门答腊 + 加里曼丹范围内撒候选点，
按 CAMS 0.4° 格点去重，找出真正互不相同的空间单元。

用法：python tools/discover_nodes.py
输出：docs/node_discovery.json
"""
import json
import os
import time
import urllib.error
import urllib.request
from collections import OrderedDict

CANDIDATES = OrderedDict([
    # ---- 大吉隆坡（原 8 站点，预期塌缩为 3 格）----
    ("KL_City",        (3.107, 101.726, "KL", "受体")),
    ("Port_Klang",     (3.000, 101.392, "KL", "受体")),
    ("Sepang",         (2.828, 101.703, "KL", "受体")),
    ("Putrajaya",      (2.926, 101.696, "KL", "候选")),
    ("Batu_Muda",      (3.212, 101.682, "KL", "候选")),
    ("Petaling_Jaya",  (3.107, 101.606, "KL", "候选")),
    ("Shah_Alam",      (3.073, 101.518, "KL", "候选")),
    ("Banting",        (2.813, 101.502, "KL", "候选")),
    # ---- 马来半岛走廊 ----
    ("Seremban",       (2.729, 101.938, "Peninsula", "候选中转")),
    ("Malacca",        (2.189, 102.250, "Peninsula", "候选中转")),
    ("Muar",           (2.045, 102.568, "Peninsula", "候选中转")),
    ("Johor_Bahru",    (1.492, 103.741, "Peninsula", "受体/源")),
    ("Kuantan",        (3.807, 103.326, "Peninsula", "受体/源")),
    ("Temerloh",       (3.450, 102.417, "Peninsula", "候选中转")),
    ("Kota_Bharu",     (6.125, 102.238, "Peninsula", "受体/源")),
    ("Kuala_Terengganu", (5.330, 103.140, "Peninsula", "源")),
    ("Penang",         (5.414, 100.329, "Peninsula", "受体/源")),
    ("Alor_Setar",     (6.118, 100.368, "Peninsula", "源")),
    ("Ipoh",           (4.597, 101.090, "Peninsula", "源")),
    ("Cameron_Highland", (4.472, 101.376, "Peninsula", "源")),
    # ---- 苏门答腊火点区（跨境烟霾源头）----
    ("Banda_Aceh",     (5.548, 95.324, "Sumatra", "源")),
    ("Medan",          (3.595, 98.672, "Sumatra", "源")),
    ("Dumai",          (1.666, 101.450, "Sumatra", "源")),
    ("Pekanbaru",      (0.507, 101.447, "Sumatra", "源")),
    ("Padang",         (-0.949, 100.354, "Sumatra", "源")),
    ("Jambi",          (-1.610, 103.613, "Sumatra", "源")),
    ("Palembang",      (-2.976, 104.775, "Sumatra", "源")),
    ("Bengkulu",       (-3.800, 102.265, "Sumatra", "源")),
    # ---- 加里曼丹（西段烟霾源）----
    ("Pontianak",      (-0.026, 109.342, "Borneo", "源")),
    ("Palangkaraya",   (-2.209, 113.917, "Borneo", "源")),
    ("Banjarmasin",    (-3.319, 114.590, "Borneo", "源")),
    ("Kuching",        (1.553, 110.359, "Borneo", "源")),
    ("Bintulu",        (3.171, 113.041, "Borneo", "源")),
])

PROBE_START = "2025-07-01"
PROBE_END = "2025-07-31"


def fetch(url, retries=3):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError) as e:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))


def probe(lat, lon):
    url = (f"https://air-quality-api.open-meteo.com/v1/air-quality"
           f"?latitude={lat}&longitude={lon}&hourly=pm2_5"
           f"&start_date={PROBE_START}&end_date={PROBE_END}"
           f"&timezone=Asia%2FKuala_Lumpur")
    d = fetch(url)
    return d["hourly"]["pm2_5"], d.get("elevation")


def main():
    series = {}
    meta = {}
    for name, (lat, lon, region, role) in CANDIDATES.items():
        try:
            s, elev = probe(lat, lon)
            series[name] = s
            meta[name] = {"lat": lat, "lon": lon, "region": region, "role": role,
                          "elevation": elev,
                          "mean": round(sum(v for v in s if v is not None) / max(1, len([v for v in s if v is not None])), 2)}
            print(f"  {name:20s} {region:10s} mean={meta[name]['mean']:6.2f}")
        except Exception as e:
            print(f"  {name:20s} FAILED: {e}")
        time.sleep(0.15)

    # 按序列完全相同分组
    groups = []
    assigned = set()
    for name in series:
        if name in assigned:
            continue
        g = [name]
        assigned.add(name)
        for other in series:
            if other in assigned:
                continue
            if series[other] == series[name]:
                g.append(other)
                assigned.add(other)
        groups.append(g)

    print(f"\n=== 候选 {len(series)} 个 → 唯一格点 {len(groups)} 个 ===")
    out = {"probe_window": [PROBE_START, PROBE_END], "cells": []}
    for g in groups:
        head = g[0]
        print(f"  cell mean={meta[head]['mean']:6.2f}  {g}")
        out["cells"].append({
            "representative": head,
            "members": g,
            "mean_pm25": meta[head]["mean"],
            "lat": meta[head]["lat"],
            "lon": meta[head]["lon"],
            "region": meta[head]["region"],
            "role": meta[head]["role"],
        })

    os.makedirs("docs", exist_ok=True)
    with open("docs/node_discovery.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nwrote docs/node_discovery.json")


if __name__ == "__main__":
    main()

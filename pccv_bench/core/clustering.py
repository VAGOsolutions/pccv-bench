"""Clustering of raw detector boxes into detection groups."""

from __future__ import annotations

def _boxes_close(b1, b2, gap_px: float) -> bool:
    """True if AABBs overlap or are within gap_px of each other."""
    x1a, y1a, x2a, y2a, _ = b1
    x1b, y1b, x2b, y2b, _ = b2
    dx = max(0.0, max(x1a, x1b) - min(x2a, x2b))
    dy = max(0.0, max(y1a, y1b) - min(y2a, y2b))
    return dx <= gap_px and dy <= gap_px


def cluster_dets(dets: list, gap_px: float) -> list[list]:
    """Single-link clustering by AABB proximity. Returns list of clusters."""
    n = len(dets)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if _boxes_close(dets[i], dets[j], gap_px):
                union(i, j)

    groups: dict[int, list] = {}
    for i, d in enumerate(dets):
        r = find(i)
        groups.setdefault(r, []).append(d)
    return list(groups.values())


def merge_cluster(cluster: list) -> tuple[float, float, float, float, float]:
    x1 = min(d[0] for d in cluster)
    y1 = min(d[1] for d in cluster)
    x2 = max(d[2] for d in cluster)
    y2 = max(d[3] for d in cluster)
    s  = max(d[4] for d in cluster)
    return x1, y1, x2, y2, s

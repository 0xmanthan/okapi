```
 ██████╗ ██╗  ██╗ █████╗ ██████╗ ██╗
██╔═══██╗██║ ██╔╝██╔══██╗██╔══██╗██║
██║   ██║█████╔╝ ███████║██████╔╝██║
██║   ██║██╔═██╗ ██╔══██║██╔═══╝ ██║
╚██████╔╝██║  ██╗██║  ██║██║     ██║
 ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝
▚▚ ▚▚ ▚▚ ▚▚ ▚▚ ▚▚ ▚▚ ▚▚ ▚▚ ▚▚ ▚▚ ▚▚
  a vector database, written from scratch in Go
```

Okapi stores vectors with JSON payloads and finds the ones nearest to a query. It filters on payload fields, combines keyword and semantic ranking, survives crashes, and replicates across nodes.

It's named after **Okapi BM25**, the ranking function from City University London's Okapi retrieval system, which Okapi uses for hybrid search.

> **Status:** this README describes Okapi once all features are built. Implemented so far: exact search, distance kernels, the recall harness. Everything else is in progress.

## Features

**Search**
- **HNSW** approximate nearest-neighbour index, tunable with `M`, `ef_construction` and `ef_search`.
- **Exact search** (flat index), used as ground truth and for small or heavily filtered sets.
- **Metrics:** cosine, dot product, squared L2.
- **Filtered search:** `color = "red" AND price < 100`. A query planner brute-forces small filtered sets and walks the HNSW graph for large ones, so recall holds at any filter selectivity.
- **Hybrid search:** BM25 keyword ranking and vector ranking fused with Reciprocal Rank Fusion.

**Storage**
- **Write-ahead log:** every acknowledged write is on disk before the response returns.
- **Segments:** writes go to an in-memory segment that is sealed into an immutable, memory-mapped segment on disk.
- **Crash recovery:** on restart, sealed segments are mapped and the WAL tail is replayed.
- **Deletes and updates** via tombstones, cleaned up by background compaction.
- **Quantization:** SQ8 (4× smaller), PQ (up to 32× smaller) and binary, with full-precision rescoring.
- **Snapshots:** consistent backups you can restore on any node.

**Server**
- **gRPC and REST APIs** over the same service layer.
- **Collections**, each with its own dimension, metric and index settings.
- **Concurrent** reads and writes, checked with Go's race detector.
- **API-key authentication.**
- **Prometheus metrics** (QPS, p50/p99 latency, segment and WAL sizes) and **pprof** profiling.

**Cluster**
- **Sharding:** points are spread across shards by id.
- **Replication:** each shard is a Raft group, so a 3-node cluster keeps serving reads and writes with one node down.

## Architecture

```
            ┌──────────── API (gRPC + REST) ────────────┐
            │ collections · upsert · search · delete    │
            └──────────────────┬────────────────────────┘
                     Query planner (filter + vector + hybrid)
          ┌──────────────┬─────┴────────┬────────────────┐
     Vector index    Payload index   Text index (BM25)   Distance kernels
     (flat, HNSW,    (roaring        (merged with        (cosine/dot/L2,
      quantized)      bitmaps)        vectors via RRF)    unrolled/SIMD)
          └──────────────┴─────┬────────┴────────────────┘
               Storage: WAL → mutable segment → immutable segments
               (mmap, tombstones, compaction, snapshots)
                               │
               Cluster: sharding + Raft replication
```

## Quick start

```bash
go build -o okapi ./cmd/okapi
./okapi --data ./okapi-data --http :8080 --grpc :9090
```

Create a collection:

```bash
curl -X PUT localhost:8080/collections/products \
  -d '{"dim": 384, "metric": "cosine"}'
```

Insert points:

```bash
curl -X PUT localhost:8080/collections/products/points \
  -d '{"points": [
        {"id": 1, "vector": [0.12, -0.03, ...], "payload": {"name": "red running shoes", "color": "red", "price": 89}},
        {"id": 2, "vector": [0.08,  0.11, ...], "payload": {"name": "blue trail shoes",  "color": "blue", "price": 120}}
      ]}'
```

Search with a filter:

```bash
curl -X POST localhost:8080/collections/products/search \
  -d '{"vector": [0.10, -0.01, ...], "k": 10,
       "filter": {"must": [{"key": "color", "match": "red"},
                           {"key": "price", "range": {"lt": 100}}]}}'
```

Hybrid search (keywords + vector):

```bash
curl -X POST localhost:8080/collections/products/search \
  -d '{"vector": [0.10, -0.01, ...], "text": "red running shoes", "k": 10}'
```

### As a Go library

```go
idx := flat.New(128)
idx.Add(42, vec)                 // copies vec; error if len(vec) != 128
res, err := idx.Search(q, 10)    // []flat.Result{ID, Dist}, nearest first
```

## Benchmarks

Measured on an Intel i5-1135G7 (4 cores, 8 threads). k = 10 throughout. Numbers are filled in as each part is measured.

### Search: SIFT1M (1,000,000 × 128-d)

| Index | Settings | recall@10 | QPS | p50 | p99 | Memory |
|---|---|---|---|---|---|---|
| Flat | — | 1.0000 | — | — | — | — |
| HNSW | M=16, ef_search=50 | — | — | — | — | — |
| HNSW | M=16, ef_search=200 | — | — | — | — | — |
| HNSW + SQ8 | rescore ×4 | — | — | — | — | — |
| HNSW + PQ | m=16, rescore ×4 | — | — | — | — | — |
| hnswlib (reference) | M=16, ef_search=50 | — | — | — | — | — |

### Search: siftsmall (10,000 × 128-d)

| Index | recall@10 | QPS | p50 |
|---|---|---|---|
| Flat | 1.0000 | ~900 | ~0.9 ms |
| HNSW | — | — | — |

### Filtered search: recall@10 vs filter selectivity

| Points matching filter | recall@10 | QPS |
|---|---|---|
| 50% | — | — |
| 10% | — | — |
| 1% | — | — |

### Writes and recovery

| | Result |
|---|---|
| Inserts/sec (WAL + fsync, batch of 100) | — |
| Recovery time after kill -9, 1M points | — |
| Acked writes lost across 100 crash runs | — |

### Distance kernels (one distance, d = 128)

| | plain loop | 4-way unrolled | SIMD |
|---|---|---|---|
| L2² | 94 ns | 55 ns | — |
| Dot | 85 ns | 51 ns | — |

## Development

Needs Go 1.24+.

```bash
# SIFT small dataset (5 MB)
mkdir -p data && curl -o data/siftsmall.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
tar xzf data/siftsmall.tar.gz -C data

go test ./...                          # tests (dataset tests skip without data/)
go test -race ./...                    # with the race detector
go test -run='^$' -bench=. ./distance  # distance benchmarks
go run ./cmd/eval                      # recall@10, QPS, per-query detail
```

`cmd/eval` checks every index against exact ground truth. A change that makes search faster but lowers recall shows up there.

## Layout

```
cmd/eval/          recall@k and QPS harness
dataset/           .fvecs / .ivecs readers
distance/          L2², dot, normalize (plain reference + unrolled)
index/flat/        exact k-NN index with a bounded max-heap
internal/errs/     shared error kinds
```

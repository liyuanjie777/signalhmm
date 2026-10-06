# mapsignal

A C++ tool for nanopore current data and training hidden Markov models (HMMs).

The main workflow uses a reference-based chain model to probabilistically align multiple current sequences and estimate representative mean current levels and their emission distributions. Positions associated with the same k-mer can share an emission model, pooling information across positions and reads.

The resulting mean current profile consists of emission means estimated after HMM alignment, rather than an element-wise average of unaligned signals. With Gaussian mixture emissions, training also estimates component means, covariance matrices, and mixture weights.

## Building

Dependencies: a C++17 compiler, CMake, OpenMP, and Zstd.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The examples below use an executable named `signalhmm` with the `viewbin` and `hmm` subcommands, following the new command-line interface. If your CMake configuration still uses the original target name `MapSignal`, replace the executable name accordingly. The subcommands must also be connected to the corresponding dispatch logic in `main()`.

## 1. Binary Data Format and viewbin

### File Layout

A file contains a sequence of read records. Each record stores a UUID, a chromosome name, and an independently compressed Zstd payload.

```text
File
├── read 0: UUID + chrom length + chrom + compressed length + payload
├── read 1: UUID + chrom length + chrom + compressed length + payload
└── ...
```

The current format has no separate global file header, version number, or persistent index table. A reader can scan record headers, skip payloads using their compressed lengths, and build an index of file offsets.

Fields appear in the following order:

| Field | Storage Type | Size | Description |
| --- | --- | --- | --- |
| UUID | Character bytes | 36 bytes | Read identifier, without a terminating `\0` |
| Chrom length | 32-bit integer | 4 bytes | Byte length of the following chromosome string |
| Chrom | Character bytes | Variable | Chromosome or group name, without a terminating `\0` |
| Compressed length | 32-bit integer | 4 bytes | Byte length of the following Zstd payload |
| Payload | Binary bytes | Variable | Compressed current and moves data for this read |

After decompression, the payload has this layout:

```text
[Signal byte length: 4 bytes]
[Signal values: float32 array]
[Moves byte length: 4 bytes]
[Moves values: int32 array]
[Sequence byte length: 4 bytes]
[Sequences: char array]
```

`data` contains current values. `mv` contains signal-to-sequence position or movement information; its exact encoding is defined by the data-producing program. The file does not include an additional description of the `mv` encoding.

### Listing Read Summaries

Print a summary of every read:

```bash
./signalhmm viewbin reads.bin summary all
```

Print summaries for the first 10 reads:

```bash
./signalhmm viewbin reads.bin summary 10
```

Example summary:

```text
INDEX  UUID                                  CHROM  SIGNAL_COUNT
0      00000000-0000-0000-0000-000000000001  chr1   125000
1      00000000-0000-0000-0000-000000000002  chr1   98300
```

Indices are **zero-based** and follow the physical record order in the file, without regrouping by chromosome. Signal and moves lengths are stored inside the compressed payload, so displaying these counts requires decompressing the corresponding records.

### Inspecting a Specific Read

Print the complete contents of read index 5, the sixth read in the file:

```bash
./signalhmm viewbin reads.bin show 5
```

The output includes the read index, UUID, chromosome, and all signal and moves values. Arrays can be displayed side by side using matching array indices:

```text
Index: 5
UUID: 00000000-0000-0000-0000-000000000006
Chrom: chr1

        Signal          MV
        72.100           0
        73.250           1
        71.830           0
```

These values illustrate the output format only. If the arrays differ in length, missing entries can be displayed as `-`. Displaying the arrays side by side does not define or change the meaning of `mv`.

For long signals, redirect the output to a text file:

```bash
./signalhmm viewbin reads.bin show 5 > read5.txt
```

## 2. HMM Training

### Training Objective

Training uses expectation-maximization (EM) to optimize a model from multiple nanopore current reads. The input model specifies reference positions, state connectivity, and initial emission parameters.

Each iteration performs the following steps:

1. Compute forward and backward probabilities within the paths and candidate bands allowed by the reference model.
2. Calculate posterior weights assigning current observations to states.
3. Accumulate weighted signal statistics and update emission parameters.
4. Update transition and initial-state probabilities if enabled by the training options.

For a single Gaussian emission, the mean represents the average current level of a state or k-mer, and the variance describes its variability. A Gaussian mixture model (GMM) stores multiple components; a single representative mean can be obtained by weighting the component means by their mixture weights.

The current training entry point loads an existing model JSON file. It does not automatically construct a state graph from FASTA. The JSON fields `nodes`, `links`, and `emission` describe states, transition edges, and emission parameters, respectively. Multiple nodes can reference the same emission model to share k-mer parameters.

### Example Command

```bash
./signalhmm hmm \
    --data reads.bin \
    --model model.json \
    --out trained.json \
    --batch-size 20 \
    --sampling-number 1500 \
    --max-iter 10 \
    --max-band 20 \
    --method te
```

| Option | Description |
| --- | --- |
| `--input` | Input binary read file |
| `--model` | Input model JSON file |
| `--output` | Output path for the trained model JSON |
| `--batch-size` | Number of reads requested per batch |
| `--sampling-number` | Read-count threshold per iteration; 0 disables this threshold |
| `--iter-max` | Number of training iterations |
| `--band` | Width parameter used to construct candidate state bands from the input position mapping |
| `--method` | Parameter groups to update during training |

The sampling threshold is checked after a batch has been processed, so the actual number of processed reads can exceed it. Read-loading and filtering conditions also determine which data participate in training.

The `--method` string can combine the following characters:

| Value | Parameters Updated |
| --- | --- |
| `e` | Emission parameters |
| `t` | Transition probabilities |
| `p` | Initial-state probabilities |
| `te` | Emission and transition parameters |
| `tep` | All three parameter groups |

To focus on estimating mean current levels, start with `--method e` to keep transitions fixed. Use `te` when joint optimization is needed.

The current implementation accumulates statistics across batches and updates parameters at the end of each iteration. The batch size does not specify how frequently model parameters are updated. Training saves a model JSON file; it does not automatically export a separate mean current profile as text.

### Training Log

Example output from a 10-iteration run:

```text
HMM: 2026-10-06 15:44:27
Active Threads: 1
Iter: 0, Prob: -2.2926, Residual: -2.292577
Iter: 1, Prob: 0.2022, Residual: 2.494740
Iter: 2, Prob: 0.7135, Residual: 0.511323
Iter: 3, Prob: 0.9087, Residual: 0.195209
Iter: 4, Prob: 0.9440, Residual: 0.035352
Iter: 5, Prob: 0.9548, Residual: 0.010748
Iter: 6, Prob: 0.9594, Residual: 0.004610
Iter: 7, Prob: 0.9614, Residual: 0.002017
Iter: 8, Prob: 0.9634, Residual: 0.001938
Iter: 9, Prob: 0.9643, Residual: 0.000902
Finish: 2026-10-06 15:44:44
```

In the current `EM_step()`, forward normalization coefficients are summed over observations and divided by the observation count for each read. These per-read values are then summed across the reads used in training:

```text
Prob = sum_read [ sum_t coeff[read][t] / T_read ]
Residual = Prob_current - Prob_previous
```

Thus, `Prob` is the **sum of per-read mean log-likelihoods per observation**. It is neither a probability between 0 and 1 nor an additional average across reads. With candidate bands, the score reflects the allowed paths. Emissions for continuous current measurements are probability densities, which can exceed 1, so a positive log-likelihood score is valid.

Set `OMP_NUM_THREADS` to configure the thread count; report the actual team size from inside a parallel region when needed.

```bash
OMP_NUM_THREADS=8 ./mapsignal train \
    --input reads.bin --model model.json --out trained.json \
    --batch-size 20 --sampling-number 1500 \
    --max-iter 10 --max-band 20 --method et
```
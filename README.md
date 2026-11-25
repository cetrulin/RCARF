# RCARF (MOA 2017.06)

MOA 2017.06 implementation of **RCARF** (Recurring Concepts Adaptive Random Forest), **EvolvingRCARF**, and early **EPCH** (Evolving Pool of Classifiers with History) prototypes from the PhD thesis *Adaptive Algorithms For Classification On High-Frequency Data Streams: Application To Finance*.

- For the successor **GroCH** (MOA 2024.07 + CapyMOA + GNG), see [groch-moa](https://github.com/cetrulin/groch-moa).

## Repository Layout (MOA 2017.06)

```text
src/
└── moa/
    ├── classifiers/
    │   ├── core/driftdetection/
    │   │   ├── AbstractEvolvingChangeDetector.java
    │   │   ├── EvolvingADWIN.java
    │   │   ├── EvolvingADWINChangeDetector.java
    │   │   └── EvolvingChangeDetector.java
    │   └── meta/
    │       ├── RCARF.java
    │       ├── EvolvingRCARF.java
    │       ├── RecurringConceptsAdaptiveRandomForest.java
    │       ├── EPCHsingle.java
    │       └── EPCH.java
    └── evaluation/
        └── DynamicWindowClassificationPerformanceEvaluator.java
```

- **`RCARF`**: Main recurring-concept ensemble supporting arbitrary MOA base learners (`-l`), dynamic comparison windows, drift decision mechanisms (`-k 0|1|2`), and optional LUFO Concept History cap (`-n`).
- **`EvolvingRCARF`**: `RCARF` variant assigning heterogeneous ADWIN drift/warning sensitivities across ensemble slots via `EvolvingADWINChangeDetector`.
- **`RecurringConceptsAdaptiveRandomForest`**: Earlier ARF-only (`ARFHoeffdingTree`) version of `RCARF`.
- **`EPCHsingle` / `EPCH`**: Early GNG topology-grouped Concept History prototypes (require `moa.classifiers.igngsvm.gng.*` from [groch-moa](https://github.com/cetrulin/groch-moa) tag `v0.5-moa2017.06`; `EPCH.java` targets an older 7-arg `GNG` constructor).

## Requirements

- **MOA 2017.06** (`moa-2017.06.jar` + `weka-dev` + `sizeofag`, from [SourceForge](https://sourceforge.net/projects/moa-datastream/files/MOA/2017%20June/moa-release-2017.06b.zip/download) or Maven Central `nz.ac.waikato.cms.moa:moa:2017.06`)
- **JDK 8+**

## Build & Run

Compile all RCARF classes against MOA 2017.06:

```bash
mkdir -p build
javac -encoding ISO-8859-1 -cp "path/to/moa-2017.06.jar:path/to/lib/*" -d build \
  src/moa/evaluation/*.java \
  src/moa/classifiers/core/driftdetection/*.java \
  src/moa/classifiers/meta/RCARF.java \
  src/moa/classifiers/meta/EvolvingRCARF.java \
  src/moa/classifiers/meta/RecurringConceptsAdaptiveRandomForest.java
```

Run `RCARF` via `moa.DoTask`:

```bash
java -Xmx2g -cp "build:path/to/moa-2017.06.jar:path/to/lib/*" moa.DoTask \
  "EvaluatePrequential -l (meta.RCARF -s 10 -j 1 -k 2 -n 20 -g) -s (generators.SEAGenerator) -i 10000 -f 1000"
```

Key `RCARF` flags:
- `-l`: Base learner (default `trees.ARFHoeffdingTree -e 2000000 -g 50 -c 0.01`)
- `-s`: Ensemble size (default `10`)
- `-j`: Concurrent worker threads (`0`/`1` = single-threaded, `-1` = all cores)
- `-k`: Drift decision mechanism (`0` = always switch, `1` = false alarm if bkg not ready, `2` = also false alarm if active is best)
- `-n`: Max Concept History size with LUFO eviction (`0` = unlimited)
- `-r`: Disable recurring concept handling (standard ARF behavior)
- `-g`: Disable event log file export

## Notes on Reproducibility

- Commit `fdbdc49` preserves **bit-exact parity** with the original repository while fixing thread-safety, static `ConceptHistory` collisions, and edge-case crashes.
- Commit `4faba24` applies behavior-changing bug fixes (deep concept copying, `findGroup` min/scaling, dynamic window accuracy-vs-error math, `EvolvingADWINChangeDetector` reset, direct Euclidean NN, and LUFO eviction).

# RCARF (MOA 2017.06)

This repo holds the MOA 2017.06 code for **RCARF** (Recurring Concepts Adaptive Random Forest), **EvolvingRCARF**, and the early **EPCH** (Evolving Pool of Classifiers with History) prototypes from my PhD thesis *Adaptive Algorithms For Classification On High-Frequency Data Streams: Application To Finance*.

If you're looking for the newer **GroCH** implementation (MOA 2024.07, CapyMOA, and GNG), head over to [groch-moa](https://github.com/cetrulin/groch-moa).

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

- **`RCARF`**: The main recurring-concept ensemble. It's base-learner agnostic (`-l`), uses dynamic comparison windows, supports drift decision modes (`-k 0|1|2`), and lets you cap Concept History with LUFO (`-n`).
- **`EvolvingRCARF`**: Same setup as `RCARF`, except it gives each ensemble slot a different ADWIN sensitivity through `EvolvingADWINChangeDetector`.
- **`RecurringConceptsAdaptiveRandomForest`**: An older ARF-only (`ARFHoeffdingTree`) version of `RCARF` that I've kept here for reference.
- **`EPCHsingle` / `EPCH`**: Early GNG topology-grouped Concept History prototypes. You'll need `moa.classifiers.igngsvm.gng.*` from [groch-moa](https://github.com/cetrulin/groch-moa) tag `v0.5-moa2017.06` to compile `EPCHsingle` (`EPCH.java` targets an older 7-arg `GNG` constructor, so it won't compile as-is against that tag).

## Requirements

- **MOA 2017.06** (`moa-2017.06.jar`, `weka-dev`, and `sizeofag` from [SourceForge](https://sourceforge.net/projects/moa-datastream/files/MOA/2017%20June/moa-release-2017.06b.zip/download) or Maven Central `nz.ac.waikato.cms.moa:moa:2017.06`)
- **JDK 8+**

## Build & Run

Compile the RCARF classes against MOA 2017.06:

```bash
mkdir -p build
javac -encoding ISO-8859-1 -cp "path/to/moa-2017.06.jar:path/to/lib/*" -d build \
  src/moa/evaluation/*.java \
  src/moa/classifiers/core/driftdetection/*.java \
  src/moa/classifiers/meta/RCARF.java \
  src/moa/classifiers/meta/EvolvingRCARF.java \
  src/moa/classifiers/meta/RecurringConceptsAdaptiveRandomForest.java
```

Run `RCARF` with `moa.DoTask`:

```bash
java -Xmx2g -cp "build:path/to/moa-2017.06.jar:path/to/lib/*" moa.DoTask \
  "EvaluatePrequential -l (meta.RCARF -s 10 -j 1 -k 2 -n 20 -g) -s (generators.SEAGenerator) -i 10000 -f 1000"
```

Main `RCARF` flags you'll use:
- `-l`: Base learner (default `trees.ARFHoeffdingTree -e 2000000 -g 50 -c 0.01`)
- `-s`: Ensemble size (default `10`)
- `-j`: Worker threads (`0`/`1` = single-threaded, `-1` = all cores)
- `-k`: Drift decision mode (`0` = always switch, `1` = false alarm if bkg isn't ready, `2` = also false alarm if active is best)
- `-n`: Max Concept History size with LUFO eviction (`0` = unlimited)
- `-r`: Turn off recurring concept handling (falls back to standard ARF)
- `-g`: Turn off event log file export

## Tests

You don't need JUnit to run the regression checks. Just compile and run with `-ea`:

```bash
javac -encoding ISO-8859-1 -cp "build:path/to/moa-2017.06.jar:path/to/lib/*" -d build tests/RCARFTest.java
java -ea -cp "build:path/to/moa-2017.06.jar:path/to/lib/*" RCARFTest
```

## Notes on Reproducibility

- If you're reproducing the original numbers, check out `fdbdc49`. It keeps bit-exact parity while fixing thread-safety, static `ConceptHistory` collisions, and edge-case crashes.
- If you want the bug fixes that do change model output, use `4faba24` or later. That commit fixes deep concept copying, `findGroup` min/scaling, dynamic window accuracy-vs-error math, `EvolvingADWINChangeDetector` reset, direct Euclidean NN, and LUFO eviction.

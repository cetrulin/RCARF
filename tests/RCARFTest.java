import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.core.driftdetection.EvolvingADWINChangeDetector;
import moa.classifiers.meta.EvolvingRCARF;
import moa.classifiers.meta.RCARF;
import moa.classifiers.meta.RecurringConceptsAdaptiveRandomForest;
import moa.core.InstanceExample;
import moa.evaluation.DynamicWindowClassificationPerformanceEvaluator;
import moa.streams.generators.SEAGenerator;

/**
 * Self-contained regression test suite (run with {@code java -ea RCARFTest}).
 */
public class RCARFTest {

    private static Instance[] recurringStream(int n) {
        SEAGenerator gen = new SEAGenerator();
        gen.functionOption.setValue(1);
        gen.instanceRandomSeedOption.setValue(42);
        gen.prepareForUse();
        Instance[] data = new Instance[n];
        for (int i = 0; i < n; i++) {
            Instance inst = (Instance) gen.nextInstance().getData().copy();
            if (((i / 500) % 2) == 1) {
                inst.setClassValue(1.0 - inst.classValue());
            }
            data[i] = inst;
        }
        return data;
    }

    public static void main(String[] args) {
        Instance[] data = recurringStream(2000);
        InstancesHeader hdr = new InstancesHeader(data[0].dataset());

        // 1. EvolvingADWINChangeDetector: pre-input delta access & change flag reset
        EvolvingADWINChangeDetector det = new EvolvingADWINChangeDetector(0.01);
        assert Math.abs(det.getDelta() - 0.01) < 1e-12 : "getDelta before input failed";
        det.setDelta(0.05);
        assert Math.abs(det.getDelta() - 0.05) < 1e-12 : "setDelta before input failed";
        boolean sawChange = false, clearedAfterChange = false;
        for (int i = 0; i < 200; i++) det.input(0.0);
        for (int i = 0; i < 200; i++) {
            det.input(1.0);
            if (det.getChange()) sawChange = true;
            else if (sawChange) clearedAfterChange = true;
        }
        assert sawChange : "Expected ADWIN change detection";
        assert clearedAfterChange : "isChangeDetected stayed stuck true";
        det.resetLearning();
        assert !det.getChange() : "resetLearning did not clear isChangeDetected";

        // 2. DynamicWindowClassificationPerformanceEvaluator: preserves pre-added models & error policy direction
        DynamicWindowClassificationPerformanceEvaluator ev =
                new DynamicWindowClassificationPerformanceEvaluator(10, 1, 5, 0.2, 0.65, true, 0, 0, "test");
        ev.reset();
        ev.addModel(1, 0.2, 10);
        double[] correctVote = new double[]{0.0, 0.0};
        correctVote[(int) data[0].classValue()] = 1.0;
        ev.addResult(new InstanceExample(data[0]), correctVote);
        assert ev.containsIndex(0) && ev.containsIndex(1) : "First addResult wiped pre-added model";
        assert ev.getWindowSize(0) == 11 : "Expected window growth (error 0.0 <= priorError 0.2), got " + ev.getWindowSize(0);

        // 3. EvolvingRCARF with ensembleSize < 5 (no divide-by-zero)
        EvolvingRCARF ev3 = new EvolvingRCARF();
        ev3.ensembleSizeOption.setValue(3);
        ev3.numberOfJobsOption.setValue(1);
        ev3.disableEventsLogFileOption.setValue(true);
        ev3.setModelContext(hdr);
        ev3.prepareForUse();
        ev3.resetLearning();
        for (Instance inst : data) {
            ev3.getVotesForInstance(inst.copy());
            ev3.trainOnInstance(inst.copy());
        }

        // 4. Per-instance ConceptHistory isolation (two interleaved RCARF instances) + LUFO cap
        RCARF a = new RCARF();
        RCARF b = new RCARF();
        for (RCARF r : new RCARF[]{a, b}) {
            r.ensembleSizeOption.setValue(4);
            r.numberOfJobsOption.setValue(1);
            r.maxConceptHistorySizeOption.setValue(2);
            r.disableEventsLogFileOption.setValue(true);
            r.warningDetectionMethodOption.setValueViaCLIString("ADWINChangeDetector -a 0.01");
            r.driftDetectionMethodOption.setValueViaCLIString("ADWINChangeDetector -a 0.001");
            r.setModelContext(hdr);
            r.prepareForUse();
            r.resetLearning();
        }
        for (Instance inst : data) {
            a.trainOnInstance(inst.copy());
            b.trainOnInstance(inst.copy());
        }

        // 5. Multi-threaded RCARF (-j 4) + executor cleanup + legacy RecurringConceptsAdaptiveRandomForest smoke
        RCARF mt = new RCARF();
        mt.ensembleSizeOption.setValue(6);
        mt.numberOfJobsOption.setValue(4);
        mt.disableEventsLogFileOption.setValue(true);
        mt.warningDetectionMethodOption.setValueViaCLIString("ADWINChangeDetector -a 0.01");
        mt.driftDetectionMethodOption.setValueViaCLIString("ADWINChangeDetector -a 0.001");
        mt.setModelContext(hdr);
        mt.prepareForUse();
        mt.resetLearning();
        for (Instance inst : data) {
            mt.getVotesForInstance(inst.copy());
            mt.trainOnInstance(inst.copy());
        }
        mt.resetLearning();

        RecurringConceptsAdaptiveRandomForest legacy = new RecurringConceptsAdaptiveRandomForest();
        legacy.ensembleSizeOption.setValue(3);
        legacy.numberOfJobsOption.setValue(1);
        legacy.disableEventsLogFileOption.setValue(true);
        legacy.setModelContext(hdr);
        legacy.prepareForUse();
        legacy.resetLearning();
        for (int i = 0; i < 200; i++) legacy.trainOnInstance(data[i].copy());

        System.out.println("ALL RCARF TESTS PASSED");
        System.exit(0);
    }
}

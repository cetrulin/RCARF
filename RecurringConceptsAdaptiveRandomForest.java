/*
 *    AdaptiveRandomForest.java
 * 
 *    @author Heitor Murilo Gomes (heitor_murilo_gomes at yahoo dot com dot br)
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 */
package moa.classifiers.meta;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.DoubleVector;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;

import moa.classifiers.trees.ARFHoeffdingTree;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.DynamicWindowClassificationPerformanceEvaluator;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import moa.classifiers.core.driftdetection.ChangeDetector;


/**
 * Adaptive Random Forest
 *
 * <p>Adaptive Random Forest (ARF). The 3 most important aspects of this 
 * ensemble classifier are: (1) inducing diversity through resampling;
 * (2) inducing diversity through randomly selecting subsets of features for 
 * node splits (See moa.classifiers.trees.ARFHoeffdingTree.java); (3) drift 
 * detectors per base tree, which cause selective resets in response to drifts. 
 * It also allows training background trees, which start training if a warning
 * is detected and replace the active tree if the warning escalates to a drift. </p>
 *
 * <p>See details in:<br> Heitor Murilo Gomes, Albert Bifet, Jesse Read, 
 * Jean Paul Barddal, Fabricio Enembreck, Bernhard Pfharinger, Geoff Holmes, 
 * Talel Abdessalem. Adaptive random forests for evolving data stream classification. 
 * In Machine Learning, DOI: 10.1007/s10994-017-5642-8, Springer, 2017.</p>
 *
 * <p>Parameters:</p> <ul>
 * <li>-l : Classiï¬�er to train. Must be set to ARFHoeffdingTree</li>
 * <li>-s : The number of trees in the ensemble</li>
 * <li>-o : How the number of features is interpreted (4 options): 
 * "Specified m (integer value)", "sqrt(M)+1", "M-(sqrt(M)+1)"</li>
 * <li>-m : Number of features allowed considered for each split. Negative 
 * values corresponds to M - m</li>
 * <li>-a : The lambda value for bagging (lambda=6 corresponds to levBag)</li>
 * <li>-j : Number of threads to be used for training</li>
 * <li>-x : Change detector for drifts and its parameters</li>
 * <li>-p : Change detector for warnings (start training bkg learner)</li>
 * <li>-w : Should use weighted voting?</li>
 * <li>-u : Should use drift detection? If disabled then bkg learner is also disabled</li>
 * <li>-q : Should use bkg learner? If disabled then reset tree immediately</li>
 * </ul>
 *
 * @author Heitor Murilo Gomes (heitor_murilo_gomes at yahoo dot com dot br)
 * @version $Revision: 1 $
 */
public class RecurringConceptsAdaptiveRandomForest extends AbstractClassifier implements MultiClassClassifier {

    @Override
    public String getPurposeString() {
        return "Recurring Concepts Adaptive Random Forest algorithm for evolving data streams from Suarez-Cetrulo et al.";
    }
    
    private static final long serialVersionUID = 1L;

    public ClassOption treeLearnerOption = new ClassOption("treeLearner", 'l',
            "Random Forest Tree.", ARFHoeffdingTree.class,
            "ARFHoeffdingTree -e 2000000 -g 50 -c 0.01");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
        "The number of trees.", 10, 1, Integer.MAX_VALUE);
    
    public MultiChoiceOption mFeaturesModeOption = new MultiChoiceOption("mFeaturesMode", 'o', 
        "Defines how m, defined by mFeaturesPerTreeSize, is interpreted. M represents the total number of features.",
        new String[]{"Specified m (integer value)", "sqrt(M)+1", "M-(sqrt(M)+1)",
            "Percentage (M * (m / 100))"},
        new String[]{"SpecifiedM", "SqrtM1", "MSqrtM1", "Percentage"}, 1);
    
    public IntOption mFeaturesPerTreeSizeOption = new IntOption("mFeaturesPerTreeSize", 'm',
        "Number of features allowed considered for each split. Negative values corresponds to M - m", 2, Integer.MIN_VALUE, Integer.MAX_VALUE);
    
    public FloatOption lambdaOption = new FloatOption("lambda", 'a',
        "The lambda parameter for bagging.", 6.0, 1.0, Float.MAX_VALUE);

    public IntOption numberOfJobsOption = new IntOption("numberOfJobs", 'j',
        "Total number of concurrent jobs used for processing (-1 = as much as possible, 0 = do not use multithreading)", 1, -1, Integer.MAX_VALUE);
    
    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'x',
        "Change detector for drifts and its parameters", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-5");

    public ClassOption warningDetectionMethodOption = new ClassOption("warningDetectionMethod", 'p',
        "Change detector for warnings (start training bkg learner)", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-4");
    
    public FlagOption disableWeightedVote = new FlagOption("disableWeightedVote", 'w', 
            "Should use weighted voting?");
    
    public FlagOption disableDriftDetectionOption = new FlagOption("disableDriftDetection", 'u',
        "Should use drift detection? If disabled then bkg learner is also disabled");

    public FlagOption disableBackgroundLearnerOption = new FlagOption("disableBackgroundLearner", 'q', 
        "Should use bkg learner? If disabled then reset tree immediately.");
    
	// ////////////////////////////////////////////////
	// ADDED IN RCARF by @suarezcetrulo
	// ////////////////////////////////////////////////
    public FlagOption disableRecurringDriftDetectionOption = new FlagOption("disableRecurringDriftDetection", 'r', 
            "Should save old learners to compare against in the future? If disabled then recurring concepts are not handled explicitely.");
    
    public FlagOption rememberConceptWindowOption = new FlagOption("rememberConceptWindow", 'i', 
            "Should remember last window size when retrieving a concept? If disabled then retrieved concepts will have a default window size.");
    
    public IntOption defaultWindowOption = new IntOption("defaultWindow", 'd', 
            "Number of rows by default in Dynamic Sliding Windows.", 10, 1, Integer.MAX_VALUE);
    
    public IntOption windowIncrementsOption = new IntOption("windowIncrements", 'c', 
            "Size of the increments or decrements in Dynamic Sliding Windows.", 1, 1, Integer.MAX_VALUE);
    
    public IntOption minWindowSizeOption = new IntOption("minWindowSize", 'z', 
            "Minimum window size in Dynamic Sliding Windows.", 5, 1, Integer.MAX_VALUE);
    
    public IntOption windowResizePolicyOption = new IntOption("windowResizePolicy",'y', 
    		"Policy to update the size of the window. Ordered by complexity, being 0 the simplest one and 3 the one with most complexity.", 0, 0, 2);
    
    public FloatOption thresholdOption = new FloatOption("thresholdOption", 't', 
            "Decision threshold for recurring concepts (-1 = threshold option disabled).", 0.65, -1, Float.MAX_VALUE);
    
    public FlagOption resizeAllWindowsOption = new FlagOption("resizeAllWindows", 'b', 
    		"Should the comparison windows for old learners be also dynamic? ");
    		//+ "(0 = only the active model has a dynamic window, 1 = active and background models have dynamic windows, 2 = all models, "
    		//+ "including historic concepts). Window size changes in historic concepts during evaluation will only be saved "
    		//+ "if the historic model is selected as new active model and the threshold option is not disabled.", 1, 0, 2);

	// ////////////////////////////////////////////////
	// ////////////////////////////////////////////////
    protected static final int FEATURES_M = 0;
    protected static final int FEATURES_SQRT = 1;
    protected static final int FEATURES_SQRT_INV = 2;
    protected static final int FEATURES_PERCENT = 3;
    
    protected static final int SINGLE_THREAD = 0;
	
    protected ARFBaseLearner[] ensemble;
    protected long instancesSeen;
    protected int subspaceSize;
    protected BasicClassificationPerformanceEvaluator evaluator;

    private ExecutorService executor;
    
    @Override
    public void resetLearningImpl() {
        // Reset attributes
        this.ensemble = null;
        this.subspaceSize = 0;
        this.instancesSeen = 0;
        this.evaluator = new BasicClassificationPerformanceEvaluator();
        
        // Multi-threading
        int numberOfJobs;
        if(this.numberOfJobsOption.getValue() == -1) 
            numberOfJobs = Runtime.getRuntime().availableProcessors();
        else 
            numberOfJobs = this.numberOfJobsOption.getValue();
        // SINGLE_THREAD and requesting for only 1 thread are equivalent. 
        // this.executor will be null and not used...
        if(numberOfJobs != AdaptiveRandomForest.SINGLE_THREAD && numberOfJobs != 1)
            this.executor = Executors.newFixedThreadPool(numberOfJobs);
    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {
        ++this.instancesSeen;
        if(this.ensemble == null) 
            initEnsemble(instance);
        
        Collection<TrainingRunnable> trainers = new ArrayList<TrainingRunnable>();
        for (int i = 0 ; i < this.ensemble.length ; i++) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(instance));
            InstanceExample example = new InstanceExample(instance);
            // Aqui añade ejemplo a evaluador
            this.ensemble[i].evaluator.addResult(example, vote.getArrayRef());
            // Aqui añade ejemplo a evaluador interno
            // TODO: if warning window open
            if(this.ensemble[i].internalWindowEvaluator!=null) this.ensemble[i].internalWindowEvaluator.addResult(example, vote.getArrayRef());
            // Aqui añade ejemplo a evaluador interno de los old concepts 
            // todo:this.ensemble[i].evaluator.addResult(example, vote.getArrayRef());
            
    		/*
    		LIKE THIS (ONLY AS EXAMPLE)
    		// Add error to main classifier window
    		this.bkgLearner.internalWindowEvaluator.add((long) this.bkgLearner.evaluator.getPerformanceMeasurements()[1].getValue()); // convert in error. still accuracy percent
    		if(ConceptHistory.backgroundDynamicWindowsFlag>=1) this.bkgLearner.internalWindowEvaluator.evaluator.windowResize((((ConceptHistory.decisionThreshold==-1) ? -1.0 : this.errorBeforeWarning))); // we add error of main classifier in all the classifiers (bkg or old)
    		
    		// Dynamic windows for old models
    		for (ConceptLearner auxConcept : oldLearners.getConceptHistoryValues()) {
    			auxConcept.conceptLearner.internalWindowEvaluator.add((long) auxConcept.conceptLearner.evaluator.getPerformanceMeasurements()[1].getValue()); // Convert in error. still accuracy percent
    			if (ConceptHistory.backgroundDynamicWindowsFlag==2) auxConcept.conceptLearner.evaluator.windowResize((((ConceptHistory.decisionThreshold==-1) ? -1.0 : this.errorBeforeWarning)));
    		}*/
            
            
            int k = MiscUtils.poisson(this.lambdaOption.getValue(), this.classifierRandom);
            if (k > 0) {
                if(this.executor != null) {
                    TrainingRunnable trainer = new TrainingRunnable(this.ensemble[i], 
                        instance, k, this.instancesSeen);
                    trainers.add(trainer);
                }
                else { // SINGLE_THREAD is in-place... 
                    this.ensemble[i].trainOnInstance(instance, k, this.instancesSeen);
                }
            }
        }
        if(this.executor != null) {
            try {
                this.executor.invokeAll(trainers);
            } catch (InterruptedException ex) {
                throw new RuntimeException("Could not call invokeAll() on training threads.");
            }
        }
    }

    @Override
    public double[] getVotesForInstance(Instance instance) {
        Instance testInstance = instance.copy();
        if(this.ensemble == null) 
            initEnsemble(testInstance);
        DoubleVector combinedVote = new DoubleVector();

        for(int i = 0 ; i < this.ensemble.length ; ++i) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(testInstance));
            if (vote.sumOfValues() > 0.0) {
                vote.normalize();
                double acc = this.ensemble[i].evaluator.getPerformanceMeasurements()[1].getValue();
                if(! this.disableWeightedVote.isSet() && acc > 0.0) {                        
                    for(int v = 0 ; v < vote.numValues() ; ++v) {
                        vote.setValue(v, vote.getValue(v) * acc);
                    }
                }
                combinedVote.addValues(vote);
            }
        }
        return combinedVote.getArrayRef();
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public void getModelDescription(StringBuilder arg0, int arg1) {
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    
    protected void initEnsemble(Instance instance) {
    	
        // Init the ensemble.
        int ensembleSize = this.ensembleSizeOption.getValue();
        this.ensemble = new ARFBaseLearner[ensembleSize];
        
        // TODO: this should be an option with default = BasicClassificationPerformanceEvaluator
//      BasicClassificationPerformanceEvaluator classificationEvaluator = (BasicClassificationPerformanceEvaluator) getPreparedClassOption(this.evaluatorOption);
        BasicClassificationPerformanceEvaluator classificationEvaluator = new BasicClassificationPerformanceEvaluator();
        
        this.subspaceSize = this.mFeaturesPerTreeSizeOption.getValue();
  
        // The size of m depends on:
        // 1) mFeaturesPerTreeSizeOption
        // 2) mFeaturesModeOption
        int n = instance.numAttributes()-1; // Ignore class label ( -1 )
        
        switch(this.mFeaturesModeOption.getChosenIndex()) {
            case AdaptiveRandomForest.FEATURES_SQRT:
                this.subspaceSize = (int) Math.round(Math.sqrt(n)) + 1;
                break;
            case AdaptiveRandomForest.FEATURES_SQRT_INV:
                this.subspaceSize = n - (int) Math.round(Math.sqrt(n) + 1);
                break;
            case AdaptiveRandomForest.FEATURES_PERCENT:
                // If subspaceSize is negative, then first find out the actual percent, i.e., 100% - m.
                double percent = this.subspaceSize < 0 ? (100 + this.subspaceSize)/100.0 : this.subspaceSize / 100.0;
                this.subspaceSize = (int) Math.round(n * percent);
                break;
        }
        // Notice that if the selected mFeaturesModeOption was 
        //  AdaptiveRandomForest.FEATURES_M then nothing is performed in the
        //  previous switch-case, still it is necessary to check (and adjusted) 
        //  for when a negative value was used. 
        
        // m is negative, use size(features) + -m
        if(this.subspaceSize < 0)
            this.subspaceSize = n + this.subspaceSize;
        // Other sanity checks to avoid runtime errors. 
        //  m <= 0 (m can be negative if this.subspace was negative and 
        //  abs(m) > n), then use m = 1
        if(this.subspaceSize <= 0)
            this.subspaceSize = 1;
        // m > n, then it should use n
        if(this.subspaceSize > n)
            this.subspaceSize = n;
        
        ARFHoeffdingTree treeLearner = (ARFHoeffdingTree) getPreparedClassOption(this.treeLearnerOption);
        treeLearner.resetLearning();
        
        for(int i = 0 ; i < ensembleSize ; ++i) {
            treeLearner.subspaceSizeOption.setValue(this.subspaceSize);
            this.ensemble[i] = new ARFBaseLearner(
                i, 
                (ARFHoeffdingTree) treeLearner.copy(), 
                (BasicClassificationPerformanceEvaluator) classificationEvaluator.copy(), 
                this.instancesSeen, 
                ! this.disableBackgroundLearnerOption.isSet(),
                ! this.disableDriftDetectionOption.isSet(), 
                driftDetectionMethodOption,
                warningDetectionMethodOption,
                false,
                ! this.disableRecurringDriftDetectionOption.isSet(),
                false, // @suarezcetrulo : first model is not old.
                new Window(this.defaultWindowOption.getValue(), this.windowIncrementsOption.getValue(), this.minWindowSizeOption.getValue(), this.thresholdOption.getValue(), 
        				this.rememberConceptWindowOption.isSet()? true: false, this.resizeAllWindowsOption.isSet()? true: false, windowResizePolicyOption.getValue()),
                null // @suarezcetrulo : Windows start at NULL (till the first earning is reached)
                );
        }
    }
    
    /**
     * Inner class that represents a single tree member of the forest. 
     * It contains some analysis information, such as the numberOfDriftsDetected, 
     */
    protected final class ARFBaseLearner {
        public int indexOriginal;
        public long createdOn;
        public long lastDriftOn;
        public long lastWarningOn;
        public ARFHoeffdingTree classifier;
        public boolean isBackgroundLearner;
        public boolean isOldLearner;
        
        // The drift and warning object parameters. 
        protected ClassOption driftOption;
        protected ClassOption warningOption;
        
        // Drift and warning detection
        protected ChangeDetector driftDetectionMethod;
        protected ChangeDetector warningDetectionMethod;
        
        public boolean useBkgLearner;
        public boolean useDriftDetector;
        public boolean useRecurringLearner; // @suarezcetrulo

        
        // Bkg learner
        protected ARFBaseLearner bkgLearner;
        
        // Statistics
        public BasicClassificationPerformanceEvaluator evaluator;
        protected int numberOfDriftsDetected;
        protected int numberOfWarningsDetected;
        
		// ////////////////////////////////////////////////
		// ADDED IN RCARF by @suarezcetrulo
		// ////////////////////////////////////////////////
        
        // Internal statistics
        protected DynamicWindowClassificationPerformanceEvaluator internalWindowEvaluator; // only used in background and old classifiers
        protected double lastError;
        protected double errorBeforeWarning;
        protected Window windowProperties;

        // Recurring learner
        protected ARFBaseLearner bestRecurringLearner;
        protected boolean recurringConceptDetected; 
      
        // Running learners from the concept history. 
        // The concept history is a concurrent array list that can be changed at any moment by any model, as they work in parallel.
        // Therefore, for the sake of simplicity, the snapshot below of the concept history is taken at the start of the warning window.
        // A recurring concept will only be eligible from this list if they still exist in the concept history 
        protected ConceptHistoryLearners historySnapshot;	
        
        // Copy of main model at the beginning of the warning window for its copy in the Concept History
        protected Concept tmpCopyOfModel;  
        // ////////////////////////////////////////////////
		// ////////////////////////////////////////////////

        private void init(int indexOriginal, ARFHoeffdingTree instantiatedClassifier, BasicClassificationPerformanceEvaluator evaluatorInstantiated, 
            long instancesSeen, boolean useBkgLearner, boolean useDriftDetector, ClassOption driftOption, ClassOption warningOption, boolean isBackgroundLearner, 
            boolean useRecurringLearner, boolean isOldLearner, Window windowProperties, DynamicWindowClassificationPerformanceEvaluator internalEvaluator) { // last parameters added by @suarezcetrulo
            this.indexOriginal = indexOriginal;
            this.createdOn = instancesSeen;
            this.lastDriftOn = 0;
            this.lastWarningOn = 0;
            
            this.classifier = instantiatedClassifier;
            this.evaluator = evaluatorInstantiated;
            this.internalWindowEvaluator = internalEvaluator; // only used in bkg and retrieved old models
            
            this.useBkgLearner = useBkgLearner;
            this.useRecurringLearner = useRecurringLearner;
            this.useDriftDetector = useDriftDetector;
            
            this.numberOfDriftsDetected = 0;
            this.numberOfWarningsDetected = 0;
            this.isBackgroundLearner = isBackgroundLearner;

            // Window params
            this.windowProperties=windowProperties;
            // If window size inheritance is enabled, get the previous model window size. Otherwise reset it to its defaultvalue.
            this.windowProperties.setWindowSize(((windowProperties.rememberWindowSize) ? 
            										windowProperties.getWindowSize(): windowProperties.windowDefaultSize)); //TODO: get window size from internalEvaluator?
                        
            // Recurring drifts
            this.isOldLearner = isOldLearner;
            this.recurringConceptDetected = false;
            this.historySnapshot = null;
            
            // Initialize error of the new active model in the current concepts. The error obtained in previous executions is not relevant anymore.
            // This error is used for dynamic window resizing in the Dynamic Evaluator during the warning window. They'll have a real value by then.
            this.lastError = this.errorBeforeWarning  = 50.0; 

            if(this.useDriftDetector) {
                this.driftOption = driftOption;
                this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftOption)).copy();
            }

            // Init Drift Detector for Warning detection. 
            if(this.useBkgLearner) {
                this.warningOption = warningOption;
                this.warningDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.warningOption)).copy();
            }
        }

        // last inputs parameters added by @suarezcetrulo
        public ARFBaseLearner(int indexOriginal, ARFHoeffdingTree instantiatedClassifier, BasicClassificationPerformanceEvaluator evaluatorInstantiated, 
                    long instancesSeen, boolean useBkgLearner, boolean useDriftDetector, ClassOption driftOption, ClassOption warningOption, 
                    boolean isBackgroundLearner, boolean useRecurringLearner, boolean isOldLearner, 
                    Window windowProperties, DynamicWindowClassificationPerformanceEvaluator bkgInternalEvaluator) {
            init(indexOriginal, instantiatedClassifier, evaluatorInstantiated, instancesSeen, useBkgLearner, 
            		 useDriftDetector, driftOption, warningOption, isBackgroundLearner, useRecurringLearner,  isOldLearner, 
            		 windowProperties,bkgInternalEvaluator);
        }

        public void reset() {
			System.out.println("RESET IN MODEL #"+this.indexOriginal);
			// 1 Compare DT results using Window method and pick the best one
			if(this.useRecurringLearner && this.historySnapshot != null)  compareModels();
			
			// 2 Transition to the best bkg or retrieved old learner
        		if (this.recurringConceptDetected && this.bestRecurringLearner != null) { //&& this.useRecurringLearner (condition implicit in the others)
	            System.out.println("RECURRING DRIFT RESET IN MODEL #"+this.indexOriginal+" TO MODEL #"+this.bestRecurringLearner.indexOriginal);       

	            // 3 Move copy of active model made before warning to Concept History
	            ConceptHistory.historyList.put(tmpCopyOfModel.index, tmpCopyOfModel); // TODO. save a copy/clone
	            
	            // 4 New active model is the best retrieved old model
                this.classifier = this.bestRecurringLearner.classifier;
                this.driftDetectionMethod = this.bestRecurringLearner.driftDetectionMethod;
                this.warningDetectionMethod = this.bestRecurringLearner.warningDetectionMethod;
                this.evaluator = this.bestRecurringLearner.evaluator;
                this.createdOn = this.bestRecurringLearner.createdOn;
                this.windowProperties=this.bestRecurringLearner.windowProperties;
                
                // 5 Clear remove background and old learners 
                cleanWarningWindow ();
        		} 
        		else if(this.useBkgLearner && this.bkgLearner != null) {  
    	            System.out.println("DRIFT RESET IN MODEL #"+this.indexOriginal+" TO NEW MODEL #"+this.bkgLearner.indexOriginal); 
    	            // 3 Move copy of active model made before warning to Concept History
    	            ConceptHistory.historyList.put(tmpCopyOfModel.index, tmpCopyOfModel); // TODO. save a copy/clone
    	            
    	            // 4 New active model is the background model
                this.classifier = this.bkgLearner.classifier;
                this.driftDetectionMethod = this.bkgLearner.driftDetectionMethod;
                this.warningDetectionMethod = this.bkgLearner.warningDetectionMethod;
                this.windowProperties=this.bkgLearner.windowProperties;
                this.evaluator = this.bkgLearner.evaluator;
                this.createdOn = this.bkgLearner.createdOn;
                
                // 5 Clear remove background and old learners 
                cleanWarningWindow();
            }
            else { //TODO: is this the false alarm? or where is it?
                this.classifier.resetLearning();
                this.createdOn = instancesSeen;
                this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftOption)).copy();
            }
            this.evaluator.reset();
    		    // TODO: We also need to reset, clear and remove the BKG AND OLD Learners here

        }

        public void trainOnInstance(Instance instance, double weight, long instancesSeen) {
            Instance weightedInstance = (Instance) instance.copy();
            weightedInstance.setWeight(instance.weight() * weight);
            // Dont train this if its an old model (from concept history)
            if (!this.isOldLearner) this.classifier.trainOnInstance(weightedInstance);
            // Dont train bkg model if it doesnt exist
            if(this.bkgLearner != null) this.bkgLearner.classifier.trainOnInstance(instance);
            
            // Should it use a drift detector? Also, is it a backgroundLearner? If so, then do not "incept" another one. 
            if(this.useDriftDetector && !this.isBackgroundLearner) {
                boolean correctlyClassifies = this.classifier.correctlyClassifies(instance);

                // Check for warning only if useBkgLearner is active
                if(this.useBkgLearner) {       
                    /*********** warning detection ***********/
                    // Update the WARNING detection method
                    this.warningDetectionMethod.input(correctlyClassifies ? 0 : 1);
                    // Check if there was a change 
                    if(this.warningDetectionMethod.getChange()) {                			                        
                        startWarningWindow(instancesSeen);
                    } else {
                    		this.lastError = this.evaluator.getFractionIncorrectlyClassified(); // TODO: should this be in the same iteration of the warning? 
                    		// TODO: should we also copy the current classifier in here? (to have the copy before raising a warning)
                    } /*********** ************* ***********/
                } /*********** drift detection ***********/
                // Update the DRIFT detection method
                this.driftDetectionMethod.input(correctlyClassifies ? 0 : 1);
                // Check if there was a change
                if(this.driftDetectionMethod.getChange()) {
                    this.lastDriftOn = instancesSeen;
                    this.numberOfDriftsDetected++;
                    this.reset();
                } /*********** ************* ***********/
                // TODO. If its false alarm, we should set oldLearners and bgkLearners as NULL.
                // Need to understand what happens with false Alarms
            }
        }

		// ////////////////////////////////////////////////
		// ADDED IN RCARF by  @suarezcetrulo
		// ////////////////////////////////////////////////
        
        // Clean warning window (removes all learners)
        public void cleanWarningWindow () {
            // Reset concept history
            historySnapshot.resetHistory();
            historySnapshot = null;
            
            // Reset recurring concept
            this.bestRecurringLearner = null; // only a double check. 
            this.recurringConceptDetected = false; // only a double check.
            
            // Reset bkg evaluator
            this.bkgLearner = null;
            this.internalWindowEvaluator = null; // added by @suarezcetrulo
        }
        
        // Starts Warning window
        public void startWarningWindow(long instancesSeen) {
            this.lastWarningOn = instancesSeen;
            this.numberOfWarningsDetected++;
        	
        		// 0 Save error before warning
            this.errorBeforeWarning = this.lastError;

            // 1 Create background Model
            createBkgModel();

            // If we search explicitly for recurring concepts (parameter activated), then we need to deal with a concept history.
            // Therefore, we need to start the warning window and save a copy of the current classifier to be saved in the history in case of drift.
            if(this.useRecurringLearner) {
    				// 2 Temporally copy the current classifier in a concept object (it still will be in use until the Drift is confirmed). 
            		saveActiveModelOnWarning();
    				// 3 Create concept learners of the old classifiers to compare against the bkg model
    				retrieveOldModels(); 	
            } System.out.println("WARNING ACTIVE IN MODEL #"+this.indexOriginal+"with background model running #"+bkgLearner.indexOriginal);
            
            // Update the warning detection object for the current object 
            // (this effectively resets changes made to the object while it was still a bkg learner). 
            this.warningDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.warningOption)).copy();
        }
        
        
        // Saves a backup of the active model that raised a warning to be stored in the concept history in case of drift.
        public void saveActiveModelOnWarning() {
	    		this.tmpCopyOfModel.reset();
	    		this.tmpCopyOfModel = new Concept(indexOriginal, (ARFHoeffdingTree) this.classifier.copy(), this.createdOn, 
	    						(long) this.evaluator.getPerformanceMeasurements()[0].getValue(), instancesSeen, this.windowProperties);
	    		// Add the model accumulated error (from the start of the model) from the iteration before the warning
	    		this.tmpCopyOfModel.setErrorBeforeWarning(errorBeforeWarning);
	    		// A simple concept to be stored in the concept history that doesn't have a running learner.
	    		// This doesn't train. It keeps the model as it was at the beginning of the training window to be stored in case of drift.
        }
        
        // Creates BKG Model in warning window
        public void createBkgModel() {
            // Create a new bkgTree classifier
            ARFHoeffdingTree bkgClassifier = (ARFHoeffdingTree) this.classifier.copy();
            bkgClassifier.resetLearning();
                                    
            // Resets the evaluator
            BasicClassificationPerformanceEvaluator bkgEvaluator = (BasicClassificationPerformanceEvaluator) this.evaluator.copy();
            bkgEvaluator.reset();
            
            // Adding also internal evaluator (window) in bkgEvaluator (by @suarezcetrulo)
            DynamicWindowClassificationPerformanceEvaluator bkgInternalWindowEvaluator = new DynamicWindowClassificationPerformanceEvaluator (
            		this.windowProperties.getWindowSize(),this.windowProperties.getWindowIncrements(),this.windowProperties.getMinWindowSize(),
            		this.lastError,this.windowProperties.getDecisionThreshold(),true,this.windowProperties.getResizingPolicy());  
            bkgInternalWindowEvaluator.reset();
            
            // Create a new bkgLearner object
            this.bkgLearner = new ARFBaseLearner(indexOriginal, bkgClassifier, bkgEvaluator, instancesSeen, 
            									   this.useBkgLearner, this.useDriftDetector, this.driftOption, 
            									   this.warningOption, true, this.useRecurringLearner, false, 
            									   this.windowProperties, bkgInternalWindowEvaluator); // added last inputs parameter by @suarezcetrulo        	
        }

        public void retrieveOldModels() {
        		this.historySnapshot = new ConceptHistoryLearners();
        		this.historySnapshot.resetHistory();
        		
        		// Dynamic windows for old models
        		for (Concept auxConcept : ConceptHistory.getConceptHistorySnapshot().values()) {
            		//if(ConceptHistory.rememberWindowFlag) auxConcept.getLastErrors().clear(); // clears content but keeps window size
            		//else auxConcept.getLastErrors().reset(); // restart content and size of the window
            		//TODO. we need to rewrite this so we basically create a new ConceptLearnerObject with its respective input parameters (inherited or not depending on the above flag)	
        			//ConceptLearnerResults auxConceptInternalWindow = new ConceptLearnerResults(this.windowSize, ConceptHistory.windowIncrements, ConceptHistory.minWindowSize, 
        			//		ConceptHistory.decisionThreshold, (DynamicWindowClassificationPerformanceEvaluator) this.bkgLearner.internalWindowEvaluator.evaluator.copy()); // TODO: is this ok? get maybe?
        			//TODO also as said: put the input parameters in this this.object and  not in the concepthistory static class
        			DynamicWindowClassificationPerformanceEvaluator auxConceptInternalWindow = 
        					new DynamicWindowClassificationPerformanceEvaluator(
        	                		this.windowProperties.getWindowSize(),this.windowProperties.getWindowIncrements(),this.windowProperties.getMinWindowSize(),
        	                		this.lastError,this.windowProperties.getDecisionThreshold(),
        	                		this.windowProperties.getDynamicWindowInOldModelsFlag(), this.windowProperties.getResizingPolicy());  	
        			auxConceptInternalWindow.reset();        			
        			 //TODO: we need to initialize all these!"
        			
            		// Creates a Background Learner for each historic concept
        			historySnapshot.pushConceptLearner(new ConceptLearner(auxConcept.getIndex(), 
        					auxConcept.getModel(), (BasicClassificationPerformanceEvaluator) this.evaluator.copy(), // we still evaluate performance of the model with the same evaluator
        					auxConcept.getNumberOfInstancesProcessedByCreation(), auxConcept.getNumberOfInstancesClassified(), 
        					auxConcept.getNumberOfInstancesSeen(), this.windowProperties, auxConceptInternalWindow, // TODO add WindowClassificationPerformanceEvaluator
        					this.useDriftDetector, this.driftOption, this.warningOption, this.useRecurringLearner, true)); // last param true as this refers to old models
        		}
        		// Dynamic Window for main model (on test and training) - it shouldn´t be necessary. as maximum get the priorError
    			//this.internalWindowEvaluator = null;    // .clear();       			

        		// Dynamic Window for Background model, it starts with the default size.
            this.bkgLearner.internalWindowEvaluator = null;    // .clear(); 
            this.bkgLearner.internalWindowEvaluator = this.internalWindowEvaluator; 
            // it was: new Window(ConceptHistory.defaultWindow, ConceptHistory.windowIncrements, ConceptHistory.minWindowSize, ConceptHistory.decisionThreshold);
        }
                
        // Rank of concept history windows and make decision against bkg model
        public void compareModels() {
        		HashMap<Integer, Double> ranking = new HashMap<Integer, Double>();
         
        		// 1 - Add old models
	    		for (ConceptLearner auxConcept : historySnapshot.getConceptHistoryValues()) {
	    			// Check that the concept still is in the concept history and available to be selected. If so, it adds its result to the ranking
	    			if(ConceptHistory.historyList.containsKey(auxConcept.getIndex())) ranking.put(auxConcept.getIndex(), auxConcept.conceptLearner.internalWindowEvaluator.getFractionIncorrectlyClassified());
	    		}
	    		// 2 Compare this against the background model 
	    		if(Collections.min(ranking.values())>=bkgLearner.internalWindowEvaluator.getFractionIncorrectlyClassified()){
	        		this.recurringConceptDetected=true;
	        		this.bestRecurringLearner=historySnapshot.getConceptLearner(getMinKey(ranking));
	    		} else this.recurringConceptDetected=false;
        }
        
        // Aux method for getting the best classifier in a hashMap of (int modelIndex, double averageErrorInWindow) 
        private Integer getMinKey(Map<Integer, Double> map, Integer... keys) {
        	    Integer minKey = null;
            double minValue = Double.MAX_VALUE;
            for(Integer key : keys) {
                double value = map.get(key);
                if(value < minValue) {
                    minValue = value;
                    minKey = key;
                }
            } return minKey;
        }
        // ////////////////////////////////////////////////
 
        public double[] getVotesForInstance(Instance instance) {
            DoubleVector vote = new DoubleVector(this.classifier.getVotesForInstance(instance));
            return vote.getArrayRef();
        }
    }
    
    /***
     * Inner class to assist with the multi-thread execution. 
     */
    protected class TrainingRunnable implements Runnable, Callable<Integer> {
        final private ARFBaseLearner learner;
        final private Instance instance;
        final private double weight;
        final private long instancesSeen;

        public TrainingRunnable(ARFBaseLearner learner, Instance instance, 
                double weight, long instancesSeen) {
            this.learner = learner;
            this.instance = instance;
            this.weight = weight;
            this.instancesSeen = instancesSeen;
        }

        @Override
        public void run() {
            learner.trainOnInstance(this.instance, this.weight, this.instancesSeen);
        }

        @Override
        public Integer call() throws Exception {
            run();
            return 0;
        }
    }
    
    
    //Concept_history = (list of concept_representations)
    // Static and concurrent for all DTs that run in parallel
    public static class ConceptHistory {

    		// Concurrent Concept History List
    		public static ConcurrentHashMap<Integer,Concept> historyList = new ConcurrentHashMap<Integer,Concept> ();
    		
    		/*// Dynamic Sliding window properties
    		public static boolean rememberWindowFlag;
    		public static int defaultWindow;
    		public static int windowIncrements;
    		public static int minWindowSize;
    		public static double decisionThreshold;
    		public static int backgroundDynamicWindowsFlag;*/

    		
    		public ConceptHistory(){
        		historyList=new ConcurrentHashMap<Integer,Concept> ();
        		System.out.println("Concept History created");
        }
        
		public void resetHistory(){
	    		historyList.clear();
	    		historyList=new ConcurrentHashMap<Integer,Concept> ();
	    		System.out.println("Concept History reset");
        }
    
        // It adds old model to Concept History
        public void pushConcept(int index, ARFHoeffdingTree classifier, long createdOn, long classifiedInstances, long instancesSeen, Window internalWindowProperties) {        		
        		historyList.put(index,new Concept(index, classifier, createdOn, classifiedInstances, instancesSeen, internalWindowProperties));
        }
                
        public Concept removeConcept(int modelID){
    			return historyList.remove(modelID);
        }
	        
        // Getters
        
        public static HashMap<Integer,Concept> getConceptHistorySnapshot() {
			return new HashMap<Integer,Concept>(historyList);
        }
        
        public Concept pullConcept(int modelID){
    			return historyList.get(modelID);
        }
        
        public int getHistorySize() {
        		return historyList.size();
        }
        
       /* // Setters (for flags)
        public static void setRememberWindowFlag(boolean parameter) {
	    		rememberWindowFlag=parameter;
		}
        
        public static void setDefaultWindowValue(int value) {
	    		defaultWindow=value;
		}
        
        public static void setWindowIncrementsValue(int value) {
        		windowIncrements=value;
		}
        
        public static void setMinWindowSizeValue(int value) {
        		minWindowSize=value;
		}
        
        public static void setThresholdValue(double value) {
    			decisionThreshold=value;
        }
        
        public static void setBackgroundDynamicWindowsFlag(int value) {
        		backgroundDynamicWindowsFlag=value;
        }*/
        
    }
    
    public class ConceptHistoryLearners {
		protected HashMap<Integer,ConceptLearner> historyList;
		
		// Constructor
	    public ConceptHistoryLearners(HashMap<Integer,ConceptLearner> historyList){
	    		this.historyList=historyList;
	    		System.out.println("Image of Concept History created");
	    }
	    
        public ConceptHistoryLearners() {
        		this.historyList = new HashMap<Integer,ConceptLearner> ();
		}

        public void pushConceptLearner(ConceptLearner obj) {        		
    			historyList.put(obj.getIndex(),obj);
        }
        
		public Collection<ConceptLearner> getConceptHistoryValues() {
			return this.historyList.values();
        }
		
		public ConceptLearner getConcept(int key) {
			return historyList.get(key);
		}
		
	    public ARFBaseLearner getConceptLearner(int key) {
			return historyList.get(key).getBaseLearner();
	    }
				
        public void resetHistory(){
	    		historyList.clear();
	    		historyList=new HashMap<Integer,ConceptLearner> ();
	    		System.out.println("Concept History reset");
        }
    }
    
    //Concept_representation = (model, last_weight, last_used_timestamp, conceptual_vector)
    public class ConceptLearner {

    		// Concept attributes
    		protected int index;
    		protected ARFHoeffdingTree classifier;
    		protected long createdOn;
    		protected long classifiedInstances;
    		protected long instancesSeen;
    		protected ARFBaseLearner conceptLearner;
    		
   	    // Constructor
	    public ConceptLearner(int index, ARFHoeffdingTree classifier, BasicClassificationPerformanceEvaluator evaluator, 
	    						long createdOn, long classifiedInstances, long instancesSeen,
	    						Window windowProperties, DynamicWindowClassificationPerformanceEvaluator auxConceptInternalWindow, 
	    						boolean useDriftDetector, ClassOption driftOption, ClassOption warningOption, 
	    						boolean useRecurringLearner, boolean isOldModel){
	    	
	    		this.index=index;
	    		this.classifier=classifier;
	    		this.createdOn=createdOn;
	    		this.instancesSeen=instancesSeen;
	    		this.classifiedInstances=classifiedInstances;
	    		
	    		BasicClassificationPerformanceEvaluator conceptEvaluator = evaluator;
	    		conceptEvaluator.reset();       
	    		
	    		// TODO: is it ok sending the param this way?
	    		DynamicWindowClassificationPerformanceEvaluator internalConceptEvaluator = auxConceptInternalWindow; 
    			internalConceptEvaluator.reset(); // TODO: is reset ok? it should...  
    			
    			// TODO: change window properties? as latest error
            
	    		this.conceptLearner = new ARFBaseLearner(index, classifier, conceptEvaluator, instancesSeen, 
	                    false, useDriftDetector, driftOption, warningOption, true, useRecurringLearner, isOldModel, windowProperties, internalConceptEvaluator);
	    }
	    
	    public int getIndex() {
			return index;	    	
	    }
	    
	    public ARFBaseLearner getBaseLearner() {
			return conceptLearner;	    	
	    }   
	}
    
    
    
    // Window-related parameters for classifier internal comparisons during the warning window 
    public class Window{
    	
	    	// Window properties
	    	int windowSize;   
	    	int windowDefaultSize;   
	    	int windowIncrements;
	    	int minWindowSize;
	    	double decisionThreshold;
	    	int windowResizePolicy;
	    	boolean backgroundDynamicWindowsFlag;
	    	boolean rememberWindowSize;	    	

		public Window(int windowSize, int windowIncrements, int minWindowSize, 
					  double decisionThreshold, boolean rememberWindowSize, 
					  boolean backgroundDynamicWindowsFlag, int windowResizePolicy) {
			this.windowSize=windowSize;
			this.windowDefaultSize=windowSize; // the default size of a window could change overtime if there is window sizw inheritance enabled
			this.windowIncrements=windowIncrements;
			this.minWindowSize=minWindowSize;
			this.decisionThreshold=decisionThreshold;
			this.backgroundDynamicWindowsFlag=backgroundDynamicWindowsFlag;
			this.windowResizePolicy=windowResizePolicy;
			this.rememberWindowSize=rememberWindowSize;
		}
		
		
		public int getWindowSize() {
			return windowSize;
		}

		public void setWindowSize(int windowSize) {
			this.windowSize = windowSize;
		}

		public int getWindowIncrements() {
			return windowIncrements;
		}

		public void setWindowIncrements(int windowIncrements) {
			this.windowIncrements = windowIncrements;
		}

		public int getMinWindowSize() {
			return minWindowSize;
		}

		public void setMinWindowSize(int minWindowSize) {
			this.minWindowSize = minWindowSize;
		}

		public double getDecisionThreshold() {
			return decisionThreshold;
		}

		public void setDecisionThreshold(double decisionThreshold) {
			this.decisionThreshold = decisionThreshold;
		}
		
		public int getResizingPolicy() {
			return windowResizePolicy;
		}

		public void setResizingPolicy(int value) {
			this.windowResizePolicy = value;
		}
		
		public boolean getDynamicWindowInOldModelsFlag() {
			return backgroundDynamicWindowsFlag;
		}

		public void getDynamicWindowInOldModelsFlag(boolean flag) {
			this.backgroundDynamicWindowsFlag = flag;
		}
		
    }
    
    // A concept object that describes a model and the meta-data associated to this in the concept history.
    public static class Concept {

		// Concept attributes
		protected int index;
		protected ARFHoeffdingTree classifier;
		protected long createdOn;
		protected long classifiedInstances;
		protected long instancesSeen;
		protected Window internalWindowProperties;
		
		protected double errorBeforeWarning;
	
		public Concept(){
			// Default concept constructor
	    }
			
	    // Constructor
	    public Concept(int index, ARFHoeffdingTree classifier,
					  long createdOn, long classifiedInstances, long instancesSeen, 
					  Window internalWindowProperties){
	    		this.index=index;
	    		this.classifier=classifier;
	    		this.createdOn=createdOn;
	    		this.internalWindowProperties=internalWindowProperties;
	    		this.instancesSeen=instancesSeen;
	    		this.classifiedInstances=classifiedInstances;
	    }
	    
	    public void reset() {
	    		classifier.resetLearning();
	    		classifier = null;
	    		internalWindowProperties = null;
	    		classifiedInstances = instancesSeen = createdOn = -1;
		}
	    
	    // Getters
	
		public ARFHoeffdingTree getModel() {
	    		return classifier;
	    }
	    
	    public int getIndex() {
			return index;	    	
	    }
		
	    public long getNumberOfInstancesProcessedByCreation() {
			return createdOn;	    	
	    }
		
	    public long getNumberOfInstancesSeen() {
			return instancesSeen;	    	
	    }
	    
	    public long getNumberOfInstancesClassified() {
			return classifiedInstances;	    	
	    }
	    
	    public double getErrorBeforeWarning() {
			return this.errorBeforeWarning;
	    }
	    
	    // Setters
	    
	    public void setErrorBeforeWarning(double value) {
	    		this.errorBeforeWarning=value;
	    }
	    
	}
    
}

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
import com.sun.org.apache.xalan.internal.xsltc.compiler.sym;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;

import moa.classifiers.trees.ARFHoeffdingTree;
import moa.evaluation.BasicClassificationPerformanceEvaluator;

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
 * <li>-l : Classiﬁer to train. Must be set to ARFHoeffdingTree</li>
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
    
    public FloatOption thresholdOption = new FloatOption("thresholdOption", 't', 
            "Decision threshold for recurring concepts (-1 = threshold option disabled).", 0.65, -1, Float.MAX_VALUE);
    
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
            this.ensemble[i].evaluator.addResult(example, vote.getArrayRef());
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

	// ////////////////////////////////////////////////
	// ADDED IN RCARF by @suarezcetrulo
	// ////////////////////////////////////////////////
    // Set Window related flag to static concept history object to avoid adding them to every method.
    // These are static and global, and should not change during the execution.
    protected void setRCARFFlags() {
    		 ConceptHistory.setRememberWindowFlag(this.rememberConceptWindowOption.isSet()? true: false);   
    		 ConceptHistory.setDefaultWindowValue(this.defaultWindowOption.getValue());    	
    		 ConceptHistory.setWindowIncrementsValue(this.windowIncrementsOption.getValue());    	
    		 ConceptHistory.setMinWindowSizeValue(this.minWindowSizeOption.getValue());  
    		 ConceptHistory.setThresholdValue(this.thresholdOption.getValue());
    }
	// ////////////////////////////////////////////////
	// ////////////////////////////////////////////////
    
    protected void initEnsemble(Instance instance) {
    	
    	
    		setRCARFFlags();
    	
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
        
        // The drift and warning object parameters. 
        protected ClassOption driftOption;
        protected ClassOption warningOption;
        
        // Drift and warning detection
        protected ChangeDetector driftDetectionMethod;
        protected ChangeDetector warningDetectionMethod;
        
        public boolean useBkgLearner;
        public boolean useDriftDetector;
        
        // Bkg learner
        protected ARFBaseLearner bkgLearner;
        
        // Statistics
        public BasicClassificationPerformanceEvaluator evaluator;
        protected int numberOfDriftsDetected;
        protected int numberOfWarningsDetected;
        
		// ////////////////////////////////////////////////
		// ADDED IN RCARF by @suarezcetrulo
		// ////////////////////////////////////////////////
        public boolean useRecurringLearner; // also added to constructor and methods
        protected boolean recurringConceptDetected;
        protected Window lastErrors;
        protected ARFBaseLearner bestRecurringLearner;
        protected Concept tmpModel;
        protected ConcurrentHashMap<Integer, Concept> oldModels;
        protected boolean isOldLearner;
        protected ConceptHistoryLearners oldLearners;
        protected long lastError;
        protected long errorBeforeWarning;
		// ////////////////////////////////////////////////
		// ////////////////////////////////////////////////

        private void init(int indexOriginal, ARFHoeffdingTree instantiatedClassifier, BasicClassificationPerformanceEvaluator evaluatorInstantiated, 
            long instancesSeen, boolean useBkgLearner, boolean useDriftDetector, ClassOption driftOption, ClassOption warningOption, 
            boolean isBackgroundLearner, boolean useRecurringLearner, boolean isOldLearner, Window lastErrors) { // last parameters added by @suarezcetrulo
            this.indexOriginal = indexOriginal;
            this.createdOn = instancesSeen;
            this.lastDriftOn = 0;
            this.lastWarningOn = 0;
            
            this.classifier = instantiatedClassifier;
            this.evaluator = evaluatorInstantiated;
            this.useBkgLearner = useBkgLearner;
            this.useRecurringLearner = useRecurringLearner;
            this.useDriftDetector = useDriftDetector;
            
            this.numberOfDriftsDetected = 0;
            this.numberOfWarningsDetected = 0;
            this.isBackgroundLearner = isBackgroundLearner;

	    		// ////////////////////////////////////////////////
	    		// ADDED IN RCARF by @suarezcetrulo
	    		// ////////////////////////////////////////////////
            oldModels = ConceptHistory.historyList;
            this.isOldLearner = isOldLearner;
            this.recurringConceptDetected = false;
            this.lastErrors = lastErrors;
            this.lastError = 100; // we assume that the error before the first execution when a DT is new is 100% (worst case)
            this.errorBeforeWarning = 100;
            
	    		// ////////////////////////////////////////////////
	    		// ////////////////////////////////////////////////

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
                    boolean isBackgroundLearner, boolean useRecurringLearner, boolean isOldLearner, Window lastErrors) {
            init(indexOriginal, instantiatedClassifier, evaluatorInstantiated, instancesSeen, useBkgLearner, 
            		 useDriftDetector, driftOption, warningOption, isBackgroundLearner, useRecurringLearner,  isOldLearner, lastErrors);
        }

        public void reset() {
	    		// @suarezcetrulo
	    		// ////////////////////////////////////////////////
	    		// ADDED IN RCARF
			// ////////////////////////////////////////////////
			// 2 Move to the best BKG Learner (pick in Drift step 1)	
        		if (this.recurringConceptDetected && this.bestRecurringLearner != null) {
	            System.out.println("RECURRING DRIFT RESET IN MODEL #"+this.indexOriginal+" TO MODEL #"+this.bestRecurringLearner.indexOriginal); //added by @suarezcetrulo            		
                this.classifier = this.bestRecurringLearner.classifier;
                this.driftDetectionMethod = this.bestRecurringLearner.driftDetectionMethod;
                this.warningDetectionMethod = this.bestRecurringLearner.warningDetectionMethod;
                this.evaluator = this.bestRecurringLearner.evaluator;
                this.createdOn = this.bestRecurringLearner.createdOn;
                this.lastErrors = this.bestRecurringLearner.lastErrors;
                this.bestRecurringLearner = null;
                this.bkgLearner = null;
                //this.isOldLearner = false; this shouldnt be necessary
        		} // ////////////////////////////////////////////////
        		else if(this.useBkgLearner && this.bkgLearner != null) {  
    	            System.out.println("DRIFT RESET IN MODEL #"+this.indexOriginal+" TO NEW MODEL #"+this.bkgLearner.indexOriginal); //added by @suarezcetrulo         
                this.classifier = this.bkgLearner.classifier;
                this.driftDetectionMethod = this.bkgLearner.driftDetectionMethod;
                this.warningDetectionMethod = this.bkgLearner.warningDetectionMethod;
                this.evaluator = this.bkgLearner.evaluator;
                this.createdOn = this.bkgLearner.createdOn;
                this.lastErrors = this.bkgLearner.lastErrors; // added by @suarezcetrulo
                this.bestRecurringLearner = null;
                this.bkgLearner = null;
            }
            else {
                this.classifier.resetLearning();
                this.createdOn = instancesSeen;
                this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftOption)).copy();
            }
            this.evaluator.reset();
        }

        public void trainOnInstance(Instance instance, double weight, long instancesSeen) {
            Instance weightedInstance = (Instance) instance.copy();
            weightedInstance.setWeight(instance.weight() * weight);
            
            if (!this.isOldLearner) //Don't train if this is an old model (from concept history)
            		this.classifier.trainOnInstance(weightedInstance);
            
            if(this.bkgLearner != null)
                this.bkgLearner.classifier.trainOnInstance(instance);
            
            // Should it use a drift detector? Also, is it a backgroundLearner? If so, then do not "incept" another one. 
            if(this.useDriftDetector && !this.isBackgroundLearner) {
                boolean correctlyClassifies = this.classifier.correctlyClassifies(instance);
                
                // Check for warning only if useBkgLearner is active
                if(this.useBkgLearner) {                    
                    // Update the warning detection method
                    this.warningDetectionMethod.input(correctlyClassifies ? 0 : 1);
                         
                    // Check if there was a change (=WARNING CASE @suarezcetrulo)
                    if(this.warningDetectionMethod.getChange()) {                			
                        this.lastWarningOn = instancesSeen;
                        this.errorBeforeWarning = this.lastError; // added by @suarezcetrulo : TODO check that its alright
                        this.numberOfWarningsDetected++;
                        
                        // Create a new bkgTree classifier
                        ARFHoeffdingTree bkgClassifier = (ARFHoeffdingTree) this.classifier.copy();
                        bkgClassifier.resetLearning();
                                                
                        // Resets the evaluator
                        BasicClassificationPerformanceEvaluator bkgEvaluator = (BasicClassificationPerformanceEvaluator) this.evaluator.copy();
                        bkgEvaluator.reset();
                        
                        // Create a new bkgLearner object
                        this.bkgLearner = new ARFBaseLearner(indexOriginal, bkgClassifier, bkgEvaluator, instancesSeen, 
                            this.useBkgLearner, this.useDriftDetector, this.driftOption, this.warningOption, true, this.useRecurringLearner, false, null); // added last inputs parameter by @suarezcetrulo

	                		// ////////////////////////////////////////////////
	                		// ADDED IN RCARF by @suarezcetrulo
	            			// ////////////////////////////////////////////////
	                    if(this.useRecurringLearner) {
		        				System.out.println("WARNING ACTIVE IN MODEL #"+this.indexOriginal+"with background model running #"+bkgLearner.indexOriginal); //@suarezcetrulo
		        				// 0 Temporally clone the current classifier in a concept object (it still will be in use until the Drift is confirmed). 
		        				tmpModel.reset();
		        				tmpModel = new Concept(indexOriginal, (ARFHoeffdingTree) this.classifier.copy(), this.createdOn, 
		        								(long) this.evaluator.getPerformanceMeasurements()[0].getValue(), instancesSeen, this.lastErrors);
		        				// 1 Start of warning window
		        				startWindowEvaluation(); 	
	                    }	// ////////////////////////////////////////////////
                        
                        // Update the warning detection object for the current object 
                        // (this effectively resets changes made to the object while it was still a bkg learner). 
                        this.warningDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.warningOption)).copy();
                    } else {
                    		this.lastError = (long) this.evaluator.getPerformanceMeasurements()[1].getValue(); // @suarezcetrulo: TODO convert to error. it's still accuracy percent
                    } // TODO. Need to understand what happens with false Alarms
                }
                
	        		// ////////////////////////////////////////////////
	        		// ADDED IN RCARF by @suarezcetrulo
	    			// ////////////////////////////////////////////////   
                // 2 Runs the function updateWindows for both historyLearners, bkg and 'this' one.
                if(this.useRecurringLearner && this.oldLearners != null) updateSlidingWindows();
             	// ////////////////////////////////////////////////
                /*********** drift detection ***********/
                
                // Update the DRIFT detection method
                this.driftDetectionMethod.input(correctlyClassifies ? 0 : 1);
                // Check if there was a change
                if(this.driftDetectionMethod.getChange()) {
                    this.lastDriftOn = instancesSeen;
                    this.numberOfDriftsDetected++;
	            		// ////////////////////////////////////////////////
	            		// ADDED IN RCARF by @suarezcetrulo
	        			// ////////////////////////////////////////////////
	        			System.out.println("DRIFT IN MODEL #"+this.indexOriginal);  //@suarezcetrulo
	        			// 1 Compare DT results using Window method and pick the best one
	        			// TODO 
	        			 runningComparisons();			
	        			// ////////////////////////////////////////////////
	        			// ////////////////////////////////////////////////
                    this.reset();
                }
                // TODO. If its false alarm, we should set oldLearners and bgkLearners as NULL

            }
        }

		// ////////////////////////////////////////////////
		// ADDED IN RCARF by  @suarezcetrulo
		// ////////////////////////////////////////////////
        public void startWindowEvaluation() {
        		this.oldLearners = new ConceptHistoryLearners();
        		this.oldLearners.resetHistory();
        		
        		// Dynamic windows for old models
        		for (Concept auxConcept : ConceptHistory.getConceptHistorySnapshot().values()) {
            		if(ConceptHistory.rememberWindowFlag) auxConcept.getLastErrors().clear(); // clears content but keeps window size
            		else auxConcept.getLastErrors().reset(); // restart content and size of the window
            			        			
            		// Creates a Background Learner for each historic concept
            		oldLearners.pushConceptLearner(new ConceptLearner(auxConcept.getIndex(), 
        					auxConcept.getModel(), (BasicClassificationPerformanceEvaluator) this.evaluator.copy(), 
        					auxConcept.getNumberOfInstancesProcessedByCreation(), auxConcept.getNumberOfInstancesClassified(), 
        					auxConcept.getNumberOfInstancesSeen(), auxConcept.getLastErrors(), 
        					this.useDriftDetector, this.driftOption, this.warningOption, this.useRecurringLearner, true)); // last param true as this refers to old models
        		}
        		// Dynamic Window for main model (on test and training)
    			this.lastErrors.clear();        			
        		this.lastErrors = new Window(((ConceptHistory.rememberWindowFlag && this.lastErrors != null) ? lastErrors.windowSize: ConceptHistory.defaultWindow), 
        									  ConceptHistory.windowIncrements, ConceptHistory.minWindowSize, ConceptHistory.decisionThreshold);

        		// Dynamic Window for Background model
            this.bkgLearner.lastErrors.clear();  
            this.bkgLearner.lastErrors = new Window(((ConceptHistory.rememberWindowFlag && this.lastErrors != null) ? lastErrors.windowSize: ConceptHistory.defaultWindow), 
					  ConceptHistory.windowIncrements, ConceptHistory.minWindowSize, ConceptHistory.decisionThreshold);
        }
        
        // Inserts and re-sizes
        /** comment by @suarezcetrulo
        // TREE_ID: this.indexOriginal
        // PROCESSED_INSTANCES_BEFORE_CREATING_THIS_TREE: this.createdOn
        // Classified instances: this.evaluator.getPerformanceMeasurements()[0]
        // Classifications correct (percent): this.evaluator.getPerformanceMeasurements()[1].
        // Kappa Statistic (percent): this.evaluator.getPerformanceMeasurements()[2].
        // Kappa Temporal Statistic (percent): this.evaluator.getPerformanceMeasurements()[3]
        // Kappa M Statistic (percent): this.evaluator.getPerformanceMeasurements()[4]
        System.out.println("MODEL #"+this.indexOriginal+" with background model running #"+bkgLearner.indexOriginal);**/
        public void updateSlidingWindows() {
        		// Add error to main classifier window
        		this.lastErrors.add((long) this.evaluator.getPerformanceMeasurements()[1].getValue()); // TODO. convert in error. still accuracy percent
    			// this.Window.resize(this.priorError [-1 if threshold==-1]) // priorError before warning
        		this.lastErrors.windowResize((((ConceptHistory.decisionThreshold==-1) ? -1.0 : this.errorBeforeWarning)));
        		
        		// Add error to main classifier window
        		this.bkgLearner.lastErrors.add((long) this.bkgLearner.evaluator.getPerformanceMeasurements()[1].getValue()); // TODO. convert in error. still accuracy percent
        		this.bkgLearner.lastErrors.windowResize((((ConceptHistory.decisionThreshold==-1) ? -1.0 : this.errorBeforeWarning))); // we add error of main classifier in all the classifiers (bkg or old)
        		
        		// Dynamic windows for old models
        		for (ConceptLearner auxConcept : oldLearners.getConceptHistoryValues()) {
        			auxConcept.conceptLearner.lastErrors.add((long) auxConcept.conceptLearner.evaluator.getPerformanceMeasurements()[1].getValue()); // TODO. convert in error. still accuracy percent
        			auxConcept.conceptLearner.lastErrors.windowResize((((ConceptHistory.decisionThreshold==-1) ? -1.0 : this.errorBeforeWarning)));
        		}
        }
        
        // Rank of concept history windows + bkg model
        public void runningComparisons() {
        		/*HashMap<Integer,Window> recurrenceEvaluation= new HashMap<Integer,Window>();
        		
        		for (Concept auxConcept : oldModels.values()) {
        			recurrenceEvaluation.put(auxConcept.getIndex(), new Window());
        		}*/
        		
        		// TODO:PICK BEST BETWEEN ALL RECURRING AND WINDOW
			 /* if(recurringWindowResults better than bkgLearner // pass check)
   			this.recurringConceptDetected = true;
   			best=classifier with best performance in the running concept history that still exists in the concurrent one;
   			this.bestRecurringLearner=best;
   		*/	 
        }
        
        // Replaces a tmpConcept saved at the start of the warning window in the static ConceptHistory when a drift occurs.
        public void replaceConcepts() {
        		// TODO: 
        		
        		// TODO: We also need to reset clear and remove the Learners once we have decided
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
    
    // cetrulin 29-nov-2017
    protected class Window {
        int windowSize;
        int minimumSize; 
        int defaultSize;
        int incrementsSize;
        double threshold;
        ArrayList<Double> lastErrors;
        
        // Constructor for window: threshold = -1 means that selected size update policy is 1
        public Window(int initialSize,int increments, int minSize, double threshold){
	    		this.windowSize=initialSize;
	    		this.defaultSize=initialSize;
	    		this.incrementsSize=increments;
	        	this.minimumSize=minSize;
	    		this.threshold=threshold;
	    		this.lastErrors=new ArrayList<Double>();
	    }

        // It adds errors (1-accuracy) per classified row
        public void add(double iterarionError) {
        		// Always storing extra results just in case we increment the window size
        		if(lastErrors.size()<(this.windowSize+this.incrementsSize)) {
        			this.lastErrors.add(iterarionError);
        		} else {
        			// Remove oldest
        			this.lastErrors.remove(0);
        		}
        }
        
        public void reset() {
        		this.lastErrors.clear();
        		this.windowSize=defaultSize;
        }
        
        public void clear() {
	    		this.lastErrors.clear();
        }
        
        // Returns the latest error window creating a sublist of the last errors.
        // As we store the plus last INCREMENT_SIZE errors, we need to start from the position INCREMENT_SIZE-1 (zero indexed)
        // Size of the sublist is WINDOW_SIZE 
        public ArrayList<Double> getLastWindow (){
        		return (ArrayList<Double>) this.lastErrors.subList((this.lastErrors.size()-(this.incrementsSize-1)), this.windowSize);
        }
        
        public ArrayList<Double> getSmallerWindow (){
    			return (ArrayList<Double>) this.lastErrors.subList((this.lastErrors.size()-(this.incrementsSize-1)+this.minimumSize), this.windowSize-this.minimumSize);
        }
        
        public ArrayList<Double> getLargeWindow (){
    			return this.lastErrors;
        }
    
        public void windowResize(double priorError) {
        		// size update policy 2
        		if(priorError==-1) { // [ min(error(c,w_c)) - priorError ] > threshold -> window size increases
				if(Collections.min(getLastWindow()) - priorError > this.threshold)
					this.windowSize+=this.incrementsSize;
				else // otherwise it decreases
					this.windowSize-=this.incrementsSize;
					if(this.windowSize<this.minimumSize) 
						this.windowSize=this.minimumSize;
        		} // size update policy 1
        		else { //For each c, W_c=s  , where s is an small number / For each iteration, independently of ny situations / a = active model
            		double W_a_candidate_0=Collections.min(getLastWindow());
            		double W_a_candidate_1=Collections.min(getLargeWindow ());
            		double W_a_candidate_2=Collections.min(getSmallerWindow ());
            		if ( W_a_candidate_1 > W_a_candidate_0 && W_a_candidate_1 > W_a_candidate_2 ) // Increase window size
        				this.windowSize+=this.incrementsSize;
            		else if ( W_a_candidate_2 > W_a_candidate_0 && W_a_candidate_2 > W_a_candidate_1 ) // Decrease window size
        				this.windowSize-=this.incrementsSize;
        				if(this.windowSize<this.minimumSize) 
        					this.windowSize=this.minimumSize;
            		//else System.out.println("Window size remains the same");
        		}        		
        }
    }
    
    //Concept_history = (list of concept_representations)
    // Static and concurrent for all DTs that run in parallel
    public static class ConceptHistory {

    		// Concurrent Concept History List
    		public static ConcurrentHashMap<Integer,Concept> historyList = new ConcurrentHashMap<Integer,Concept> ();
    		
    		// Dynamic Sliding window properties
    		public static boolean rememberWindowFlag;
    		public static int defaultWindow;
    		public static int windowIncrements;
    		public static int minWindowSize;
    		public static double decisionThreshold;


    	
        /** Static class - constructor not required
    		// Constructor
        public ConceptHistory(){
        		historyList=new ConcurrentHashMap<Integer,Concept> ();
        		System.out.println("Concept History created");
        }**/
    		
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
        public void pushConcept(int index, ARFHoeffdingTree classifier, long createdOn, long classifiedInstances, long instancesSeen, Window lastErrors) {        		
        		historyList.put(index,new Concept(index, classifier, createdOn, classifiedInstances, instancesSeen, lastErrors));
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
        
        // Setters (for flags)
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
        
        
        /** The next parameters can be also obtained from the old concept itself
        public ARFBaseLearner getModel(int modelID) {
			return historyList.get(modelID).getModel();
        }
        
        public Window getLastErrors(int modelID){
        		return historyList.get(modelID).getLastErrors();
        }
        
        public int getLastExampleID(int modelID){
    			return historyList.get(modelID).getLastExampleID();
        }**/
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
		
        public void resetHistory(){
	    		historyList.clear();
	    		historyList=new HashMap<Integer,ConceptLearner> ();
	    		System.out.println("Concept History reset");
        }
	    	    
	}
    
    //Concept_representation = (model, last_weight, last_used_timestamp, conceptual_vector)
    public class ConceptLearner extends Concept {

    		// Concept attributes
    		protected int index;
    		protected ARFHoeffdingTree classifier;
    		protected long createdOn;
    		protected long classifiedInstances;
    		protected long instancesSeen;
    		ARFBaseLearner conceptLearner;
    		
	    // Constructor
	    public ConceptLearner(int index, ARFHoeffdingTree classifier, BasicClassificationPerformanceEvaluator evaluator, 
	    						long createdOn, long classifiedInstances, long instancesSeen, Window lastErrors, 
	    						boolean useDriftDetector, ClassOption driftOption, ClassOption warningOption, 
	    						boolean useRecurringLearner, boolean isOldModel){
	    	
	    		this.index=index;
	    		this.classifier=classifier;
	    		this.createdOn=createdOn;
	    		this.instancesSeen=instancesSeen;
	    		this.classifiedInstances=classifiedInstances;
            BasicClassificationPerformanceEvaluator conceptEvaluator = (BasicClassificationPerformanceEvaluator) evaluator;
            conceptEvaluator.reset();                
	    		this.conceptLearner = new ARFBaseLearner(index, classifier, conceptEvaluator, instancesSeen, 
	                    false, useDriftDetector, driftOption, warningOption, true, useRecurringLearner, isOldModel, lastErrors);
	    }
	    
	    
        
	    	   
	}
    
    public static class Concept {

		// Concept attributes
		protected int index;
		protected ARFHoeffdingTree classifier;
		protected long createdOn;
		protected long classifiedInstances;
		protected long instancesSeen;
		protected Window lastErrors;
	
		public Concept(){
			// Default concept constructor
	    }
			
	    // Constructor
	    public Concept(int index, ARFHoeffdingTree classifier,long createdOn, long classifiedInstances, long instancesSeen, Window lastErrors){
	    		this.index=index;
	    		this.classifier=classifier;
	    		this.createdOn=createdOn;
	    		this.lastErrors=lastErrors;
	    		this.instancesSeen=instancesSeen;
	    		this.classifiedInstances=classifiedInstances;
	    }
	    
	    public void reset() {
	    		classifier.resetLearning();
	    		classifier = null;
	    		lastErrors.reset();
	    		lastErrors = null;
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
	    
	    public Window getLastErrors() {
			return lastErrors;
	    }
	}
    
}

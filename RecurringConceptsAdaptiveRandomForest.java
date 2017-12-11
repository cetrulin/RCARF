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
	
    protected RCARFBaseLearner[] ensemble;
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
            // 1 Testing in active model
            this.ensemble[i].evaluator.addResult(example, vote.getArrayRef());
            
            if(!disableRecurringDriftDetectionOption.isSet()) {
	            // 2 If the warning window is open, testing in background model internal evaluator (for comparison purposes) 
	            if(this.ensemble[i].bkgLearner != null && this.ensemble[i].bkgLearner.internalWindowEvaluator!=null) 
	            		this.ensemble[i].bkgLearner.internalWindowEvaluator.addResult(example, vote.getArrayRef());
	            // 3 If the concept history is ready and it contains old models, testing in each old model internal evaluator (to compare against bkg one)
	            if(this.ensemble[i].historySnapshot != null && (this.ensemble[i].historySnapshot).getNumberOfConcepts() > 0) {
		            	for (ConceptLearner oldModel : (this.ensemble[i].historySnapshot).getConceptHistoryValues()) { // TODO: test this
		            		oldModel.getBaseLearner().internalWindowEvaluator.addResult(example, vote.getArrayRef()); // TODO: test this
		            	}
	            }
            }
            
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
        this.ensemble = new RCARFBaseLearner[ensembleSize];
        
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
            this.ensemble[i] = new RCARFBaseLearner(
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
                false, // @suarezcetrulo : first model is not old. An old model (retrieved from the concept history) doesn't train at least becomes an active model again.
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
    protected final class RCARFBaseLearner {
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
        protected RCARFBaseLearner bkgLearner;
        
        // Recurring learner
        protected RCARFBaseLearner bestRecurringLearner;
        protected boolean recurringConceptDetected; 
        
        // Running learners from the concept history. 
        // The concept history is a concurrent array list that can be changed at any moment by any model, as they work in parallel.
        // Therefore, for the sake of simplicity, the snapshot below of the concept history is taken at the start of the warning window.
        // A recurring concept will only be eligible from this list if they still exist in the concept history 
        protected ConceptHistoryLearners historySnapshot;	
        
        // Copy of main model at the beginning of the warning window for its copy in the Concept History
        protected Concept tmpCopyOfModel;  
        
        // Statistics
        public BasicClassificationPerformanceEvaluator evaluator;
        protected int numberOfDriftsDetected;
        protected int numberOfWarningsDetected;

        // Internal statistics
        public DynamicWindowClassificationPerformanceEvaluator internalWindowEvaluator; // only used in background and old classifiers
        protected double lastError;
        protected double errorBeforeWarning;
        protected Window windowProperties;

        
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
        public RCARFBaseLearner(int indexOriginal, ARFHoeffdingTree instantiatedClassifier, BasicClassificationPerformanceEvaluator evaluatorInstantiated, 
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
        		if ((this.recurringConceptDetected && this.bestRecurringLearner != null) || (this.useBkgLearner && this.bkgLearner != null)) { //&& this.useRecurringLearner (condition implicit in the others)
        			RCARFBaseLearner newActiveModel = null;
        			
    	            // 3 Move copy of active model made before warning to Concept History
    	            ConceptHistory.historyList.put(tmpCopyOfModel.index, tmpCopyOfModel.copy());
    	            
	            // 4 Pick new active model (the best one selected in step 1)
	            if(this.recurringConceptDetected && this.bestRecurringLearner != null) newActiveModel = this.bestRecurringLearner; // System.out.println("RECURRING DRIFT RESET IN MODEL #"+this.indexOriginal+" TO MODEL #"+this.bestRecurringLearner.indexOriginal);   
	            else if(this.useBkgLearner && this.bkgLearner != null) newActiveModel = this.bkgLearner; // System.out.println("DRIFT RESET IN MODEL #"+this.indexOriginal+" TO NEW MODEL #"+this.bkgLearner.indexOriginal); 
	            
                // 5 Update window size in window properties depending on window size inheritance flag (entry parameter/Option)
                newActiveModel.windowProperties.setSize(((newActiveModel.windowProperties.rememberWindowSize) ? 
                		newActiveModel.internalWindowEvaluator.getWindowSize() : newActiveModel.windowProperties.windowDefaultSize)); 
	            
	            // 6 New active model is the best retrieved old model
                this.windowProperties=newActiveModel.windowProperties;
                this.classifier = newActiveModel.classifier;
                this.driftDetectionMethod = newActiveModel.driftDetectionMethod;
                this.warningDetectionMethod = newActiveModel.warningDetectionMethod;
                this.evaluator = newActiveModel.evaluator;
                this.createdOn = newActiveModel.createdOn;
                
                // 7 Clear remove background and old learners 
                cleanWarningWindow ();
        		} 
            else { //TODO: is this the false alarm? or where is it?
                this.classifier.resetLearning();
                this.createdOn = instancesSeen;
                this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftOption)).copy();
            }
            this.evaluator.reset(); // TODO: if this is due to a false alarm, we should be doing cleanWarningWindow() here and adding this line inside
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
                        this.lastWarningOn = instancesSeen;
                        this.numberOfWarningsDetected++;
                        startWarningWindow();
                    } else {
                    		this.lastError = this.evaluator.getFractionIncorrectlyClassified(); // TODO: should this be in the same iteration of the warning? 
                    		// TODO: should we also copy the current classifier in here? currently inside warning window
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

        // Clean warning window (removes all learners)
        public void cleanWarningWindow () {
            // 1 Reset bkg evaluator
            this.bkgLearner = null;
            this.internalWindowEvaluator = null; // only a double check, as it should be always null (only used in background + old concept Learners)
            
            // 2 Reset recurring concept
            this.bestRecurringLearner = null; 
            this.recurringConceptDetected = false; 
            
            // 3 Reset concept history
            this.historySnapshot.resetHistory();
            this.historySnapshot = null;
        }
        
        // Starts Warning window
        public void startWarningWindow() {
        	
        		// 1 Save error before warning
            this.errorBeforeWarning = this.lastError;

            // 2 Create background Model
            createBkgModel();

            // If we search explicitly for recurring concepts (parameter activated), then we need to deal with a concept history.
            // Therefore, we need to start the warning window and save a copy of the current classifier to be saved in the history in case of drift.
            if(this.useRecurringLearner) {
    				// 3 Temporally copy the current classifier in a concept object (it still will be in use until the Drift is confirmed). 
            		saveActiveModelOnWarning();
    				// 4 Create concept learners of the old classifiers to compare against the bkg model
    				retrieveOldModels(); 	
            } System.out.println("WARNING ACTIVE IN MODEL #"+this.indexOriginal+"with background model running #"+bkgLearner.indexOriginal);
            // Update the warning detection object for the current object 
            // (this effectively resets changes made to the object while it was still a bkg learner). 
            this.warningDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.warningOption)).copy();
        }
        
        
        // Saves a backup of the active model that raised a warning to be stored in the concept history in case of drift.
        public void saveActiveModelOnWarning() {
        		// 1 Delete previous copy (if any)
	    		this.tmpCopyOfModel.reset();	    		
	    		this.tmpCopyOfModel = new Concept(indexOriginal, (ARFHoeffdingTree) this.classifier.copy(), this.createdOn, 
	    						(long) this.evaluator.getPerformanceMeasurements()[0].getValue(), this.lastWarningOn, this.windowProperties.copy());
	    		// 2 Add the model accumulated error (from the start of the model) from the iteration before the warning
	    		this.tmpCopyOfModel.setErrorBeforeWarning(this.errorBeforeWarning);
	    		// A simple concept to be stored in the concept history that doesn't have a running learner.
	    		// This doesn't train. It keeps the model as it was at the beginning of the training window to be stored in case of drift.
        }
        
        // Creates BKG Model in warning window
        public void createBkgModel() {
            // 1 Create a new bkgTree classifier
            ARFHoeffdingTree bkgClassifier = (ARFHoeffdingTree) this.classifier.copy();
            bkgClassifier.resetLearning();
                                    
            // 2 Resets the evaluator
            BasicClassificationPerformanceEvaluator bkgEvaluator = (BasicClassificationPerformanceEvaluator) this.evaluator.copy();
            bkgEvaluator.reset();
            
            // 3 Adding also internal evaluator (window) in bkgEvaluator (by @suarezcetrulo)
            DynamicWindowClassificationPerformanceEvaluator bkgInternalWindowEvaluator = new DynamicWindowClassificationPerformanceEvaluator (
            		this.windowProperties.getSize(),this.windowProperties.getIncrements(),this.windowProperties.getMinSize(),
            		this.lastError,this.windowProperties.getDecisionThreshold(),true,this.windowProperties.getResizingPolicy());  
            bkgInternalWindowEvaluator.reset();
            
            // 4 Create a new bkgLearner object (TODO: IF I SEE REPEATED INDEXES IT MAY BE DUE TO THIS, IT'S SENDING THE SAME INDEX. WE'D HAVE TO CREATE A NEW ONE -  MAYBE USING AN STATIC PARAM IN THE CONCEPT HISTORY(?) - SEE HOW'S DONE THIS IN THE ENSEMBLE)
            this.bkgLearner = new RCARFBaseLearner(indexOriginal, bkgClassifier, bkgEvaluator, this.lastWarningOn, 
            		this.useBkgLearner, this.useDriftDetector, this.driftOption, this.warningOption, true, this.useRecurringLearner, false, 
            									   this.windowProperties, bkgInternalWindowEvaluator); // added last inputs parameter by @suarezcetrulo        	
        }

        public void retrieveOldModels() {
        		this.historySnapshot = new ConceptHistoryLearners();
        		this.historySnapshot.resetHistory();
        		
        		// 1 Get old classifiers from an snapshot of the Concept History
        		for (Concept auxConcept : ConceptHistory.getConceptHistorySnapshot().values()) {
        			
                 // 2 Resets the evaluator for each of the Concept History
                 BasicClassificationPerformanceEvaluator auxConceptEvaluator = (BasicClassificationPerformanceEvaluator) this.evaluator.copy();
                 auxConceptEvaluator.reset();
        			
        			// 3 Create an internal evaluator for each of the Concept History
        			DynamicWindowClassificationPerformanceEvaluator auxConceptInternalWindow = new DynamicWindowClassificationPerformanceEvaluator(
        				auxConcept.windowProperties.getSize(), auxConcept.windowProperties.getIncrements(), auxConcept.windowProperties.getMinSize(),
	        			this.lastError, auxConcept.windowProperties.getDecisionThreshold(),
	        			auxConcept.windowProperties.getDynamicWindowInOldModelsFlag(), auxConcept.windowProperties.getResizingPolicy());  	
        			auxConceptInternalWindow.reset();        			
        			
            		// Creates a Concept Learner for each historic concept
        			// It sends a copy of the active evaluator only for reference, as it's only used at the end if the active model transitions to the given concept.
        			historySnapshot.pushConceptLearner(new ConceptLearner(auxConcept.getIndex(), auxConcept.getModel(), auxConceptEvaluator, 
    					auxConcept.getNumberOfInstancesProcessedByCreation(), auxConcept.getNumberOfInstancesClassified(), auxConcept.getNumberOfInstancesSeen(), 
    					auxConcept.windowProperties, auxConceptInternalWindow, this.useDriftDetector, this.driftOption, this.warningOption, this.useRecurringLearner, true)); // last param true as this refers to old models
        		}
 			
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
 
        public double[] getVotesForInstance(Instance instance) {
            DoubleVector vote = new DoubleVector(this.classifier.getVotesForInstance(instance));
            return vote.getArrayRef();
        }
    }
    
    /***
     * Inner class to assist with the multi-thread execution. 
     */
    protected class TrainingRunnable implements Runnable, Callable<Integer> {
        final private RCARFBaseLearner learner;
        final private Instance instance;
        final private double weight;
        final private long instancesSeen;

        public TrainingRunnable(RCARFBaseLearner learner, Instance instance, 
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
		
	    public RCARFBaseLearner getConceptLearner(int key) {
			return historyList.get(key).getBaseLearner();
	    }
	    
	    public int getNumberOfConcepts() {
	    		return historyList.size();
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
    		
    		public RCARFBaseLearner conceptLearner;
    		
   	    // Constructor
	    public ConceptLearner(int index, ARFHoeffdingTree classifier, BasicClassificationPerformanceEvaluator conceptEvaluator,
	    						long createdOn, long classifiedInstances, long instancesSeen,
	    						Window windowProperties, DynamicWindowClassificationPerformanceEvaluator internalConceptEvaluator, 
	    						boolean useDriftDetector, ClassOption driftOption, ClassOption warningOption, 
	    						boolean useRecurringLearner, boolean isOldModel){
	    	
	    		this.index=index;
	    		this.classifier=classifier;
	    		this.createdOn=createdOn;
	    		this.instancesSeen=instancesSeen;
	    		this.classifiedInstances=classifiedInstances;
	    		
	    		// Create concept learner
			this.conceptLearner = new RCARFBaseLearner(index, classifier, conceptEvaluator, instancesSeen, 
                    false, useDriftDetector, driftOption, warningOption, true, useRecurringLearner, isOldModel, windowProperties, internalConceptEvaluator);
    }
	    
	    public int getIndex() {
			return index;	    	
	    }
	    
	    public RCARFBaseLearner getBaseLearner() {
			return conceptLearner;	    	
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
		protected Window windowProperties;
		
		protected double errorBeforeWarning;
	
		public Concept(){
			// Default concept constructor
	    }
			
	    // Constructor
	    public Concept(int index, ARFHoeffdingTree classifier,
					  long createdOn, long classifiedInstances, long instancesSeen, 
					  Window windowProperties){
	    		this.index=index;
	    		this.classifier=classifier;
	    		this.createdOn=createdOn;
	    		this.windowProperties=windowProperties;
	    		this.instancesSeen=instancesSeen;
	    		this.classifiedInstances=classifiedInstances;
	    }
	    
	    public void reset() {
	    		classifier.resetLearning();
	    		classifier = null;
	    		windowProperties = null;
	    		index = -1;
	    		classifiedInstances = instancesSeen = createdOn = -1;
		}
	    
	    public Concept copy() {
			return new Concept(this.index, this.classifier, 
							  this.createdOn, this.classifiedInstances, this.instancesSeen, 
							  this.windowProperties);
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
		
		public Window copy() {
			return new Window(this.windowSize, this.windowIncrements, this.minWindowSize, 
					  this.decisionThreshold, this.rememberWindowSize, 
					  this.backgroundDynamicWindowsFlag, this.windowResizePolicy) ;
		}
		
		
		public int getSize() {
			return this.windowSize;
		}

		public void setSize(int windowSize) {
			this.windowSize = windowSize;
		}

		public int getIncrements() {
			return this.windowIncrements;
		}

		public void setIncrements(int windowIncrements) {
			this.windowIncrements = windowIncrements;
		}

		public int getDefaultSize() {
			return this.windowDefaultSize;
		}

		public void setDefaultSize(int windowDefaultSize) {
			this.windowDefaultSize = windowDefaultSize;
		}

		
		
		public int getMinSize() {
			return this.minWindowSize;
		}

		public void setMinSize(int minWindowSize) {
			this.minWindowSize = minWindowSize;
		}

		public double getDecisionThreshold() {
			return this.decisionThreshold;
		}

		public void setDecisionThreshold(double decisionThreshold) {
			this.decisionThreshold = decisionThreshold;
		}
		
		public boolean getRememberSizeFlag(){
			return this.rememberWindowSize;
		}
		
		public void setRememberSizeFlag(boolean flag){
			this.rememberWindowSize=flag;
		}
		
		public int getResizingPolicy() {
			return this.windowResizePolicy;
		}

		public void setResizingPolicy(int value) {
			this.windowResizePolicy = value;
		}
		
		public boolean getDynamicWindowInOldModelsFlag() {
			return this.backgroundDynamicWindowsFlag;
		}

		public void getDynamicWindowInOldModelsFlag(boolean flag) {
			this.backgroundDynamicWindowsFlag = flag;
		}
		
    }
    
    
}

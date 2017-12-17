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
import java.util.Map.Entry;
import java.util.Set;
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
        
        // 1 If the concept history is ready and it contains old models, testing in each old model internal evaluator (to compare against bkg one)
        if(ConceptHistory.historyList != null && ConceptHistory.modelsOnWarning.containsValue(true) && ConceptHistory.historyList.size() > 0) {
	        	for (Concept oldModel : ConceptHistory.historyList.values()) { // TODO: test this
	            DoubleVector oldModelVote = new DoubleVector(oldModel.ConceptLearner.getVotesForInstance(instance)); // TODO. this
	            // System.out.println("Im classifier number #"+oldModel.Concept.classifier.calcByteSize()+" created on: "+oldModel.Concept.createdOn+"  and last error was:  "+oldModel.Concept.lastError);
	        		oldModel.ConceptLearner.internalWindowEvaluator.addResult(new InstanceExample(instance), oldModelVote.getArrayRef()); // TODO: test this
	        	}
        } // else System.out.println("No models on warning");
        // TODO: add print here to show how many warnings are active each X iterations. We could also record drifts. THIS IS DONE IN RESET
        
        Collection<TrainingRunnable> trainers = new ArrayList<TrainingRunnable>();
        for (int i = 0 ; i < this.ensemble.length ; i++) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(instance));
            InstanceExample example = new InstanceExample(instance);
            // 2 Testing in active model
            this.ensemble[i].evaluator.addResult(example, vote.getArrayRef());

            if(!disableRecurringDriftDetectionOption.isSet()) {
	            // 3 If the warning window is open, testing in background model internal evaluator (for comparison purposes) 
	            if(this.ensemble[i].bkgLearner != null && this.ensemble[i].bkgLearner.internalWindowEvaluator!=null 
	            		&& this.ensemble[i].bkgLearner.internalWindowEvaluator.containsIndex(this.ensemble[i].bkgLearner.indexOriginal)) {
	                 DoubleVector bkgVote = new DoubleVector(this.ensemble[i].bkgLearner.getVotesForInstance(instance)); 
	            		this.ensemble[i].bkgLearner.internalWindowEvaluator.addResult(example, bkgVote.getArrayRef());
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
        public boolean isOldLearner; // only for reference
        
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

        // Copy of main model at the beginning of the warning window for its copy in the Concept History
        protected Concept tmpCopyOfModel;  
        
        // Statistics
        public BasicClassificationPerformanceEvaluator evaluator;
        protected int numberOfDriftsDetected;
        protected int numberOfWarningsDetected;

        // Internal statistics
        public DynamicWindowClassificationPerformanceEvaluator internalWindowEvaluator; // only used in background and old classifiers
        protected double lastError;
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

    			// 1 Decrease amount of warnings in concept history and from evaluators
			ConceptHistory.modelsOnWarning.put(this.indexOriginal, false);
            if(ConceptHistory.historyList != null && ConceptHistory.historyList.size() > 0) {
		        	for (Concept oldModel : ConceptHistory.historyList.values()) {
		        		((DynamicWindowClassificationPerformanceEvaluator) 
		        			oldModel.ConceptLearner.internalWindowEvaluator).deleteModel(this.indexOriginal);
		        	}
            } System.out.println("RESET (WARNING OFF) IN MODEL #"+this.indexOriginal+". Warning flag status (activeModelPos, Flag): "+ConceptHistory.modelsOnWarning);
			
			// 2 Transition to the best bkg or retrieved old learner
        		if (this.useBkgLearner && this.bkgLearner != null) {
        		   if(this.useRecurringLearner && ConceptHistory.historyList != null && ConceptHistory.historyList.size() > 0) {
            		   // 2.1 Compare DT results using Window method and pick the best one between concept history and bkg model.
            		   // It returns the best model in the object of the bkgLearner
        			   selectNewActiveModel();
       	           // 2.2 Move copy of active model made before warning to Concept History. Its history ID will be the last one in the history (= size)
        			   // Clean the copy afterwards.
        			   this.tmpCopyOfModel.addHistoryID(ConceptHistory.nextID());
        			   ConceptHistory.historyList.put(this.tmpCopyOfModel.historyIndex, this.tmpCopyOfModel);
        			   this.tmpCopyOfModel = null;
        			   // Consideration *: This classifier is added to the concept history, but it wont be considered by other classifiers on warning until their next warning.
        			   // If it becomes necessary in terms of implementation for this concept, to be considered immediately by the other examples in warning,
        			   // we could have a HashMap in ConceptHistory with a flag saying if a given ensembleIndexPos needs to check the ConceptHistory again and add window sizes and priorError.
        		   }
                // 2.3 Update window size in window properties depending on window size inheritance flag (entry parameter/Option)
    	            this.bkgLearner.windowProperties.setSize(((this.bkgLearner.windowProperties.rememberWindowSize) ? 
                		this.bkgLearner.internalWindowEvaluator.getWindowSize(this.bkgLearner.indexOriginal) : this.bkgLearner.windowProperties.windowDefaultSize)); 
	            
	            // 2.4 New active model is the best retrieved old model
                this.windowProperties=this.bkgLearner.windowProperties; // internalEvaluator shouldnt be inherited
                this.classifier = this.bkgLearner.classifier;
                this.driftDetectionMethod = this.bkgLearner.driftDetectionMethod;
                this.warningDetectionMethod = this.bkgLearner.warningDetectionMethod;                
                this.evaluator = this.bkgLearner.evaluator;
                this.createdOn = this.bkgLearner.createdOn; 
                
                // 2.5 Clear remove background and old learners 
            		this.bkgLearner = null; 
            		this.internalWindowEvaluator = null; // only a double check, as it should be always null (only used in background + old concept Learners)
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
            
            // Training active models and background models (if they exist). Retrieved old models are not trained.
            this.classifier.trainOnInstance(weightedInstance);            
            if(this.bkgLearner != null) this.bkgLearner.classifier.trainOnInstance(instance);
            
            // Should it use a drift detector? Also, is it a backgroundLearner? If so, then do not "incept" another one. 
            if(this.useDriftDetector && !this.isBackgroundLearner) {
                boolean correctlyClassifies = this.classifier.correctlyClassifies(instance);

                // Check for warning only if useBkgLearner is active
                if(this.useBkgLearner) { 
                    /*********** warning detection ***********/
                    // Update the WARNING detection method
                    this.warningDetectionMethod.input(correctlyClassifies ? 0 : 1);
                    // Check if there was a change – @suarezcetrulo: assumption -> in case of false alarm this triggers warning again and the bkglearner gets replaced
                    // if(this.warningDetectionMethod !=null? this.warningDetectionMethod.getChange(): false) { 
                    if(this.warningDetectionMethod.getChange()) {
                        this.lastWarningOn = instancesSeen;
                        this.numberOfWarningsDetected++;
                        
	    				   // 1 Update last error and make a backup of the current classifier in a concept object (the active one will be in use until the Drift is confirmed). 
                        // As there is no false alarms explicit mechanism (bkgLeaners keep running till replaced), this has been moved here.
                        if(this.useRecurringLearner) updateBeforeWarning();
	                	   
	                	   // 2 Start warning window to create bkg learner and retrieve old models (if option enabled)
                        startWarningWindow();
                    }

                } /*********** drift detection ***********/
                // Update the DRIFT detection method
                this.driftDetectionMethod.input(correctlyClassifies ? 0 : 1);
                // Check if there was a change
                if(this.driftDetectionMethod.getChange()) {
                    this.lastDriftOn = instancesSeen;
                    this.numberOfDriftsDetected++;

	        		   //  Transition to new model
                    this.reset();
                } 
            } 
        } 
        
        // Saves a backup of the active model that raised a warning to be stored in the concept history in case of drift.
        public void updateBeforeWarning() {  
        		// 1 Update last error before warning of the active classifier
        		// This error is the total fraction of examples incorrectly classified since this model was active until now.
    			this.lastError = this.evaluator.getFractionIncorrectlyClassified();  
    			
   			// 2 Create an internal evaluator for the Concept History
   			DynamicWindowClassificationPerformanceEvaluator tmpInternalWindow = new DynamicWindowClassificationPerformanceEvaluator(
   				this.windowProperties.getSize(), this.windowProperties.getIncrements(), this.windowProperties.getMinSize(),
        			this.lastError, this.windowProperties.getDecisionThreshold(),
        			this.windowProperties.getDynamicWindowInOldModelsFlag(), this.windowProperties.getResizingPolicy(),
        			this.indexOriginal, "created for old-retrieved classifier in ensembleIndex #"+this.indexOriginal);  	
   			tmpInternalWindow.reset();        			
    			
    			// 3 Copy Base learner for Concept History in case of Drift and store it on temporal object.
    			RCARFBaseLearner tmpConcept = new RCARFBaseLearner(this.indexOriginal, 
    					(ARFHoeffdingTree) this.classifier.copy(), (BasicClassificationPerformanceEvaluator) this.evaluator.copy(), 
    					this.createdOn, this.useBkgLearner, this.useDriftDetector, this.driftOption, this.warningOption, 
    					true, this.useRecurringLearner, true, this.windowProperties.copy(), tmpInternalWindow);
    			// TODO: do I need to reset anything?   
	    		this.tmpCopyOfModel = new Concept(tmpConcept, 
	    				this.createdOn, this.evaluator.getPerformanceMeasurements()[0].getValue(), this.lastWarningOn);
	    		
	    		// 3 Add the model accumulated error (from the start of the model) from the iteration before the warning
	    		this.tmpCopyOfModel.setErrorBeforeWarning(this.lastError);
	    		// A simple concept to be stored in the concept history that doesn't have a running learner.
	    		// This doesn't train. It keeps the model as it was at the beginning of the training window to be stored in case of drift.
        }
                
        // Starts Warning window
        public void startWarningWindow() {
        		// 0 Reset warning window
            	this.bkgLearner = null; 
    	        this.internalWindowEvaluator = null;
    	        
    	        // 1 Updating objects with warning. Turns on windows flag in Concept History.
            // Also, if the concept history is ready and it contains old models, add prior estimation and window size to each concepts history learner
        		ConceptHistory.modelsOnWarning.put(this.indexOriginal, true);
            if(ConceptHistory.historyList != null && ConceptHistory.historyList.size() > 0) {
		        	for (Concept oldModel : ConceptHistory.historyList.values()) {
		        		((DynamicWindowClassificationPerformanceEvaluator) 
		        			oldModel.ConceptLearner.internalWindowEvaluator).addModel(this.indexOriginal,this.lastError,this.windowProperties.windowSize);
		        	}
            } System.out.println("WARNING ON IN MODEL #"+this.indexOriginal+". Warning flag status (activeModelPos, Flag): "+ConceptHistory.modelsOnWarning);

        		
            // 2 Create background Model
            createBkgModel();

            // Update the warning detection object for the current object 
            // (this effectively resets changes made to the object while it was still a bkg learner). 
            this.warningDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.warningOption)).copy();
        }
        
        // Creates BKG Model in warning window
        public void createBkgModel() {
            // 1 Create a new bkgTree classifier
            ARFHoeffdingTree bkgClassifier = (ARFHoeffdingTree) this.classifier.copy();
            bkgClassifier.resetLearning();
                                    
            // 2 Resets the evaluator
            BasicClassificationPerformanceEvaluator bkgEvaluator = (BasicClassificationPerformanceEvaluator) this.evaluator.copy();
            bkgEvaluator.reset();
            
            // System.out.println("------------------------------");
            // System.out.println("Create estimator for BKG model in position: "+this.indexOriginal);
            // 3 Adding also internal evaluator (window) in bkgEvaluator (by @suarezcetrulo)
            DynamicWindowClassificationPerformanceEvaluator bkgInternalWindowEvaluator = null;
            if(this.useRecurringLearner) {
                bkgInternalWindowEvaluator = new DynamicWindowClassificationPerformanceEvaluator (
                		this.windowProperties.getSize(),this.windowProperties.getIncrements(),this.windowProperties.getMinSize(),
                		this.lastError,this.windowProperties.getDecisionThreshold(),true,this.windowProperties.getResizingPolicy(), 
                		this.indexOriginal, "created for BKG classifier in ensembleIndex #"+this.indexOriginal);  
                bkgInternalWindowEvaluator.reset();
            }
            // System.out.println("------------------------------");
            
            // 4 Create a new bkgLearner object
            this.bkgLearner = new RCARFBaseLearner(indexOriginal, bkgClassifier, bkgEvaluator, this.lastWarningOn, 
            		this.useBkgLearner, this.useDriftDetector, this.driftOption, this.warningOption, true, this.useRecurringLearner, false, 
            									   this.windowProperties, bkgInternalWindowEvaluator); // added last inputs parameter by @suarezcetrulo        	
        }
     
        // Rank of concept history windows and make decision against bkg model
        public void selectNewActiveModel() {
        		HashMap<Integer, Double> ranking = new HashMap<Integer, Double>();
        		// 1 - Add old models: get each concept score for current warning model (this.indexOriginal - pos of this model with active warning in ensemble)
        		// Concept History owns only one learner per historic concept. But each learner saves all model's independent window size and priorEstimation in a HashMap.
	    		for (Concept auxConcept : ConceptHistory.historyList.values()) 
	    			// Only take into consideration Concepts sent to the Concept History after the current model raised a warning (see this consideration in reset*) 
	    			if (auxConcept.ConceptLearner.internalWindowEvaluator.containsIndex(this.indexOriginal))
		    			ranking.put(auxConcept.getHistoryIndex(), ((DynamicWindowClassificationPerformanceEvaluator) 
		    					auxConcept.ConceptLearner.internalWindowEvaluator).getFractionIncorrectlyClassified(this.indexOriginal));
		    		
	    		// Double check just in case the best concept is no longer in the concept history, or some concepts where created after the current model raised warning.
	    		// ranking = updateWithExistingConcepts(ranking);
	    		
	    		// If there are no available choices, the new active model will be the background one. Each bkg model has its own learner.
    			if(ranking.size()>0) {
		    		// 2 Compare this against the background model 
		    		if(Collections.min(ranking.values())<=((DynamicWindowClassificationPerformanceEvaluator) 
							this.bkgLearner.internalWindowEvaluator).getFractionIncorrectlyClassified(this.bkgLearner.indexOriginal)){
		        		//System.out.println(ranking.size()); // TODO: debugging
		        		System.out.println(getMinKey(ranking)); // TODO: debugging
		        		// Extracts best recurring learner form concept history. It no longer exists in the concept history
		        		this.bkgLearner=ConceptHistory.extractConcept(getMinKey(ranking));
	    	            System.out.println("RECURRING DRIFT RESET IN MODEL #"+this.indexOriginal+" TO MODEL #"+this.bkgLearner.indexOriginal);   
		    		} else {
		    			System.out.println("The minimum recurrent concept error: "+
		    					Collections.min(ranking.values())+" is not better than the bbk learner one: "+
		    					((DynamicWindowClassificationPerformanceEvaluator) 
		    							this.bkgLearner.internalWindowEvaluator).getFractionIncorrectlyClassified(this.bkgLearner.indexOriginal));
	    	            System.out.println("DRIFT RESET IN MODEL #"+this.indexOriginal+" TO NEW MODEL #"+this.bkgLearner.indexOriginal); 
		    		}
    			} else {
    				System.out.println("0 available concepts in concept history.");
    	            System.out.println("DRIFT RESET IN MODEL #"+this.indexOriginal+" TO NEW MODEL #"+this.bkgLearner.indexOriginal); 
    			}
        }
        
        /*private HashMap<Integer, Double> updateWithExistingConcepts(HashMap<Integer, Double> ranking) {
	    		if(ranking.size()>0) {
	    			int min = getMinKey(ranking);
	    			if(!ConceptHistory.historyList.containsKey(min)) {
	    				ranking.remove(min);
	    				if(ranking.size()>0) ranking = updateWithExistingConcepts(ranking);
		    		}
	    		} return ranking;
        }*/
        
        // Aux method for getting the best classifier in a hashMap of (int modelIndex, double averageErrorInWindow) 
        private Integer getMinKey(Map<Integer, Double> map) {
        	    Integer minKey = null;
        	    System.out.println("map is: "+map+" number of keys is: "+map.keySet().size()); // TODO. debugging
            double minValue = Double.MAX_VALUE;
            for(Integer key : map.keySet()) {
            		System.out.println("Key is:"+key+" with value: "+map.get(key));
                 double value = map.get(key);
                 if(value < minValue) {
                	   System.out.println("Min error is: "+ value+" with key: "+key);
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
    		public static int lastID = 0;
    		
    		// List of ensembles with an active warning used as to determine if the history list evaluators should be in use
    		public static ConcurrentHashMap<Integer,Boolean> modelsOnWarning = new ConcurrentHashMap<Integer,Boolean> ();
    		
    		public ConceptHistory(){
        		historyList=new ConcurrentHashMap<Integer,Concept> ();
        		System.out.println("Concept History created");
        }
        
		public void resetHistory(){
	    		historyList.clear();
	    		historyList=new ConcurrentHashMap<Integer,Concept> ();
	    		System.out.println("Concept History reset");
        }

	    public static RCARFBaseLearner getConcept(int key) {
			return historyList.get(key).getBaseLearner();
	    }
	    
	    public static RCARFBaseLearner extractConcept(int key) {
	    		RCARFBaseLearner aux = historyList.get(key).getBaseLearner();
	    		historyList.remove(key);
			return aux;
	    }
	    
		public Set<Entry<Integer,Concept>> getConceptHistoryEntrySet() {
			return historyList.entrySet();
        }
	        
        // Getters
        
        public static HashMap<Integer,Concept> getConceptHistorySnapshot() {
			return new HashMap<Integer,Concept>(historyList);
        }
        
        public Concept pullConcept(int modelID){
    			return historyList.get(modelID);
        }
        
        public static int nextID() {
        		return lastID++;
        }
        
        public int getHistorySize() {
        		return historyList.size();
        }
        
    }
    
    //Concept_representation = (model, last_weight, last_used_timestamp, conceptual_vector)
    public class Concept {

    		// Concept attributes
    		protected int ensembleIndex; // position that it had in the ensemble. for reference only.
    		protected int historyIndex; // id in concept history
    		
    		// Stats
    		protected long createdOn;
    		protected long instancesSeen;
    		protected double classifiedInstances;
    		protected double errorBeforeWarning;
    		
    		// Learner
    		public RCARFBaseLearner ConceptLearner;

    		// Constructor
    		public Concept (RCARFBaseLearner ConceptLearner, long createdOn, double classifiedInstances, long instancesSeen) {
    			// Extra info
	    		this.createdOn=createdOn;
	    		this.instancesSeen=instancesSeen;
	    		this.classifiedInstances=classifiedInstances;
	    		this.ensembleIndex=ConceptLearner.indexOriginal;

	    		// Learner
    			this.ConceptLearner = ConceptLearner;
    		}
    		
    		public void addHistoryID(int id){
    			this.historyIndex=id;
    		}
    		
	    	    
	    public int getEnsembleIndex() {
			return this.ensembleIndex;	    	
	    }
	    
	    public int getHistoryIndex() {
	    		return this.historyIndex;
	    }
	    
	    public RCARFBaseLearner getBaseLearner() {
			return this.ConceptLearner;	    	
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

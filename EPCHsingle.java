/*
 *    EPCH.java
 *
 *    @author Andres Leon Suarez Cetrulo (suarezcetrulo at gmail dot com)
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

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.StringOption;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import moa.core.DoubleVector;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.options.ClassOption;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.igngsvm.gng.GNG;
import moa.classifiers.igngsvm.gng.GUnit;
import moa.classifiers.meta.EPCH.Concept;
import moa.classifiers.meta.EPCH.EPCHBaseLearner;
import moa.classifiers.meta.EPCH.Topology;
import moa.classifiers.meta.EPCH.Window;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.DynamicWindowClassificationPerformanceEvaluator;
import moa.evaluation.LearningPerformanceEvaluator;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

/**
 * Evolving Pool of Classifiers with History
 *
 * @author Andres Leon Suarez Cetrulo (suarezcetrulo at gmail dot com)
 * @version $Revision: 1 $
 */
public class EPCHsingle extends AbstractClassifier implements MultiClassClassifier {

	@Override
	public String getPurposeString() {
		return "EPCH from Suarez-Cetrulo et al.";
	}

	private static final long serialVersionUID = 1L;

	/////////////
	// Options
	// -------
	public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.", Classifier.class,
			"trees.HoeffdingTree -e 1000000 -g 200 -c 0"); // default params for hoeffding trees

	public FloatOption lambdaOption = new FloatOption("lambda", 'a', "The lambda parameter for bagging.", 6.0, 1.0,
			Float.MAX_VALUE);

	public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'x',
			"Change detector for drifts and its parameters", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-5");

	public ClassOption warningDetectionMethodOption = new ClassOption("warningDetectionMethod", 'p',
			"Change detector for warnings (start training bkg learner)", ChangeDetector.class,
			"ADWINChangeDetector -a 1.0E-4");

	public FlagOption disableWeightedVote = new FlagOption("disableWeightedVote", 'w', "Should use weighted voting?");

	public FlagOption disableDriftDetectionOption = new FlagOption("disableDriftDetection", 'u',
			"Should use drift detection? If disabled then bkg learner is also disabled");

	public FlagOption disableRecurringDriftDetectionOption = new FlagOption("disableRecurringDriftDetection", 'r',
			"Should save old learners to compare against in the future? If disabled then recurring concepts are not handled explicitely.");

	public FlagOption rememberConceptWindowOption = new FlagOption("rememberConceptWindow", 'i',
			"Should remember last window size when retrieving a concept? If disabled then retrieved concepts will have a default window size.");

	public IntOption defaultWindowOption = new IntOption("defaultWindow", 'd',
			"Number of rows by default in Dynamic Sliding Windows.", 50, 1, Integer.MAX_VALUE);

	public IntOption windowIncrementsOption = new IntOption("windowIncrements", 'c',
			"Size of the increments or decrements in Dynamic Sliding Windows.", 1, 1, Integer.MAX_VALUE);

	public IntOption minWindowSizeOption = new IntOption("minWindowSize", 'z',
			"Minimum window size in Dynamic Sliding Windows.", 5, 1, Integer.MAX_VALUE);

	public IntOption windowResizePolicyOption = new IntOption("windowResizePolicy", 'y',
			"Policy to update the size of the window. Ordered by complexity, being 0 the simplest one and 3 the one with most complexity.",
			0, 0, 2);

	public FloatOption thresholdOption = new FloatOption("thresholdOption", 't',
			"Decision threshold for recurring concepts (-1 = threshold option disabled).", 0.65, -1, Float.MAX_VALUE);

	public FlagOption resizeAllWindowsOption = new FlagOption("resizeAllWindows", 'b',
			"Should the comparison windows for old learners be also dynamic?");

	public StringOption eventsLogFileOption = new StringOption("eventsLogFile", 'e',
			"File path to export events as warnings and drifts", "./EPCH_events_log.txt");

	public FlagOption disableEventsLogFileOption = new FlagOption("disableEventsLogFile", 'g',
			"Should export event logs to analyze them in the future? If disabled then events are not logged.");

	public IntOption logLevelOption = new IntOption("eventsLogFileLevel", 'h',
			"0 only logs drifts; 1 logs drifts + warnings; 2 logs every data example", 1, 0, 2);

	public ClassOption evaluatorOption = new ClassOption("baseClassifierEvaluator", 'f',
			"Classification performance evaluation method in each base classifier for voting.",
			LearningPerformanceEvaluator.class, "BasicClassificationPerformanceEvaluator");

	public IntOption driftDecisionMechanismOption = new IntOption("driftDecisionMechanism", 'k',
			"0 does not take into account the performance active base classifier explicitely, at the time of the drift; "  +
			"1 takes into consideration active classifiers", 2, 0, 2);

	public IntOption warningWindowSizeThresholdOption = new IntOption("WarningWindowSizeThreshold", 'ñ',
			"Threshold for warning window size that defines a false a alarm.", 300, 1, Integer.MAX_VALUE);

	public FloatOption distThresholdOption = new FloatOption("distThresholdOption", 'ç',
			"Max distance allowed between topologies to be considered part of the same group.", 100000, 0, Float.MAX_VALUE);
	
	public FlagOption resetTopologyInMerginOption = new FlagOption("resetTopologyInMergin", '€',
			"Should the topology be trained from scratch after a drift signal is raised?");
	
	// Options for the topology: TODO (these should come from a meta class)
	public IntOption topologyLambdaOption = new IntOption("topologyLambda", 'o', "Topology Lambda", 100);
	public IntOption maxAgeOption = new IntOption("maxAge", 'm', "MaximumAge", 200);
	public FloatOption alfaOption = new FloatOption("alfa", '$', "Alfa", 0.5);
	public FloatOption constantOption = new FloatOption("d", '&', "d", 0.995);
	public FloatOption BepsilonOption = new FloatOption("epsilonB", '@', "EpsilonB", 0.2);
	public FloatOption NepsilonOption = new FloatOption("epsilonN", 'j', "EpsilonN", 0.006);
	public IntOption stoppingCriteriaOption = new IntOption("stoppingCriteria", 'v', "Stopping criteria", 100);
	// public FloatOption stopPercentageOption = new FloatOption("stopPercentageOption", 'P',
	//		"Stopping criteria as percentage (if 0, the static stopping criteria is )", 0, 0, 100.0);
	public FlagOption classNotAsAnAttributeInTopologyOption = new FlagOption("classNotAsAnAttributeInTopology", 'q',
			"Should the class be considered as a feature in the topology?");
	//////////
	
	protected EPCHBaseLearner active;
	protected long instancesSeen;
	protected int subspaceSize;

	// Window statistics
	protected double lastError;

	// Warning and Drifts
	public long lastDriftOn;
	public long lastWarningOn;

	// Drift and warning detection
	protected ChangeDetector driftDetectionMethod;
	protected ChangeDetector warningDetectionMethod;

	protected int numberOfDriftsDetected;
	protected int numberOfWarningsDetected;

	PrintWriter eventsLogFile;
	public int logLevel;

	Topology topology;
	Topology newTopology;
	Instances W;
	ConceptHistory CH;
	int CHid;
	int groupId;
	
	///////////////////////////////////////
	//
	// TRAINING AND TESTING OF THE ENSEMBLE
	// Data Management and Prediction modules are here.
	// All other modules also are orchestrated from here.
	// -----------------------------------
	
	@Override
	public void resetLearningImpl() {
		// Reset attributes
		this.active = null;
		this.subspaceSize = 0;
		this.instancesSeen = 0;
		this.topology = null;
		this.newTopology = null;
		this.CHid = 0;
		this.groupId = 0;

		// Reset warning and drift detection related attributes
		this.lastDriftOn = 0;
		this.lastWarningOn = 0;

		this.numberOfDriftsDetected = 0;
		this.numberOfWarningsDetected = 0;

		// Init Drift Detector
		if (!this.disableDriftDetectionOption.isSet()) {
			this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy();
		}

		// Init Drift Detector for Warning detection.
		this.warningDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.warningDetectionMethodOption)).copy();
		this.W = new Instances();  // list of training examples during warning window.
	}

	/**
	 * In EPCH, this method performs the actions of the classifier manager. Thus, in
	 * this method, warning and drift detection are performed. This method also send
	 * instances to the ensemble classifiers. Or to the single active classifier if
	 * ensemble size = 1 (default).
	 * 
	 * New BKG classifiers and switching from and to CH may also need to be here.
	 *
	 * This also trains base classifiers, track warning and drifts, and orchestrate the comparisons and replacement of classifiers.
	 * Drifts and warnings are only accessible from here.
	 *
	 * 	Steps followed:
	 *  ----------------
	 * - 0 Initialization
	 * - 1 If the concept history is ready and it contains old classifiers, then test the training instance
	 * 		in each old classifier's internal evaluator to know how their errors compare against bkg one.
	 * - 2 Update error in active classifier.
	 * - 3 Update error in background classifier's internal evaluator.
	 * - 4 Train each base classifier, orchestrating drifts and switching of classifiers.
	 * - 5 Train base classifier (Lines 4-6 of algorithm)
	 * - 6 Check for drifts and warnings only if drift detection is enabled
	 * - 7.1 Check for warning only if useBkgLearner is active.
	 * - 7.1.1 Otherwise update the topology (this is done as long as there is no active warnings).
	 * - 7.2 Check for drift
	 * - 8: Log training event
	 * 
	 * The method below implements the following lines of the algorithm:
	 * - Line 1: start topology
	 * - Lines 2-3: initialize ensemble (create base classifiers) and lists.
	 * The rest of the lines of the algorithm are triggered from here using the method
	 *  'trainBaseClassifierOnInstance' (Lines 4-35).
	 * Line 4-6: ClassifierTrain(c, x, y) -> // Train c on the current instance (x, y).
	 * The rest of the lines of the algorithm are triggered from, here by:
	 * 	- warningDetection: Lines 7-21
	 *  - topology.trainOnInstanceImpl: Lines 13-15 if warning detection is disabled
	 *  - driftDetection: Lines 22-35
	 */
	@Override
	public void trainOnInstanceImpl(Instance instance) {
		++this.instancesSeen;
		init(instance); // Step 0: Initialization

		// Step 1: Update error in concept history learners
		if (!this.disableRecurringDriftDetectionOption.isSet() && this.CH != null
				&& this.CH.getWarnings().containsValue(true) && this.CH.size() > 0) {
			this.CH.updateHistoryErrors(instance);	
		} 
		// Steps 2-4: Iterate through the ensemble for following steps (active and bkg classifiers)
		updateEvaluators(instance); 
		
		// Step 5: Train base classifier (Lines 4-6)
		this.active.trainOnInstance(instance, this.instancesSeen);
		
		// Step 6: Check for drifts and warnings only if drift detection is enabled
		if (!this.disableDriftDetectionOption.isSet()) { // && !this.active.isBackgroundLearner)
			boolean correctlyClassified = this.correctlyClassifies(instance);
			// Step 7.1: Check for warning only if useBkgLearner is active. The topology gets updated either way
			detectWarning(instance, correctlyClassified);
			this.topology.trainOnInstanceImpl(instance);  // Step 7.1.1 (Lines 13-15)
			
			// Step 7.2: Check for drift
			detectDrift(correctlyClassified);
		} 
		// Step 8: Register training example in log
		if (this.eventsLogFile != null && this.logLevelOption.getValue() >= 2)
			logEvent(getTrainExampleEvent());
	}

	public void updateEvaluators(Instance instance) {
		DoubleVector vote = new DoubleVector(this.active.getVotesForInstance(instance));
		InstanceExample example = new InstanceExample(instance);
		this.active.evaluator.addResult(example, vote.getArrayRef()); // Step 2: Testing in active classifier

		if (!disableRecurringDriftDetectionOption.isSet()) { // Step 3: Update error in background classifier's
			if (this.active.bkgLearner != null && this.active.bkgLearner.internalWindowEvaluator != null
					&& this.active.bkgLearner.internalWindowEvaluator
							.containsIndex(this.active.bkgLearner.indexOriginal)) {
				DoubleVector bkgVote = new DoubleVector(this.active.bkgLearner.getVotesForInstance(instance));
				
				// Update both active and bkg classifier internal evaluators
				this.active.bkgLearner.internalWindowEvaluator.addResult(example, bkgVote.getArrayRef());
				this.active.internalWindowEvaluator.addResult(example, vote.getArrayRef());
			}
		}
	}
	
	
	@Override
	public double[] getVotesForInstance(Instance instance) {
		Instance testInstance = instance.copy();
		if (this.active == null) init(testInstance);
		DoubleVector combinedVote = new DoubleVector();
		DoubleVector vote = new DoubleVector(this.active.getVotesForInstance(testInstance));
		if (vote.sumOfValues() > 0.0) {
			vote.normalize();
			double acc = this.active.evaluator.getPerformanceMeasurements()[1].getValue();
			if (!this.disableWeightedVote.isSet() && acc > 0.0) {
				for (int v = 0; v < vote.numValues(); ++v) vote.setValue(v, vote.getValue(v) * acc);
			}
			combinedVote.addValues(vote);
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
		// TODO. add the same to RCARF, getting a sum of all warnings/drifts in the ensemble
		// (just add a loop through ensemble[i].numberOfDriftsDetected) in rcarf
        List<Measurement> measurementList = new LinkedList<Measurement>();
        measurementList.add(new Measurement("Change detected", this.numberOfDriftsDetected));
        measurementList.add(new Measurement("Warning detected", this.numberOfWarningsDetected));
        // this.numberOfDriftsDetected = 0;
        // this.numberOfWarningsDetected = 0;
        return measurementList.toArray(new Measurement[measurementList.size()]);
    }
	
	// TODO: should initEnsemble be inside the init method? or the opposite maybe?
	private void init(Instance instance) {
		if (!this.disableDriftDetectionOption.isSet() && this.topology == null) {
			this.topology = new Topology(this.topologyLambdaOption, this.alfaOption, this.maxAgeOption,
				this.constantOption, this.BepsilonOption, this.NepsilonOption, this.classNotAsAnAttributeInTopologyOption); // algorithm line 1
			this.topology.resetLearningImpl();
		} System.out.println("prototypes created in topology:" + this.topology.getNumberOfPrototypesCreated());

		if (this.active == null) { // algorithm lines 2-3
			// Init the ensemble.
			BasicClassificationPerformanceEvaluator classificationEvaluator = (BasicClassificationPerformanceEvaluator)
			      getPreparedClassOption(this.evaluatorOption);

			// Only initialize the Concept History if the handling of recurring concepts is enabled
			if (!this.disableRecurringDriftDetectionOption.isSet()) CH = new ConceptHistory(this.distThresholdOption.getValue());
			this.W = (Instances) instance.copy().dataset();
			this.W.delete();

			try { // Start events logging and print headers
				if (this.disableEventsLogFileOption.isSet()) {
					this.eventsLogFile = null;
				} else {
					this.eventsLogFile = new PrintWriter(this.eventsLogFileOption.getValue());
					logEvent(getEventHeaders());
				}
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}

			Classifier learner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
			learner.resetLearning();

			this.active = new EPCHBaseLearner(0, (Classifier) learner.copy(),
					(BasicClassificationPerformanceEvaluator) classificationEvaluator.copy(), this.instancesSeen,
					!this.disableDriftDetectionOption.isSet(), // these are still needed con the level below
					false, // isbkglearner
					!this.disableRecurringDriftDetectionOption.isSet(),
					false, // first classifier is not in the CH.
					new Window(this.defaultWindowOption.getValue(), this.windowIncrementsOption.getValue(),
							this.minWindowSizeOption.getValue(), this.thresholdOption.getValue(),
							this.rememberConceptWindowOption.isSet() ? true : false,
							this.resizeAllWindowsOption.isSet() ? true : false, windowResizePolicyOption.getValue()),
					null, // Windows start at NULL
					this.warningWindowSizeThresholdOption.getValue());
		}
	}
	
	///////////////////////////////////////
	//
	// CLASSIFIER MANAGEMENT MODULE
	// Divided into three parts:
	// - Warning and drift handling (detection)
	// - Actions in case of drift
	// - Identification of new best group, classifier and trigger switch between classifiers.
	// -----------------------------------
	
	// WARNING AND DRIFT HANDLING

	/**
	 * This method implements all the actions that happen when the warning detection is enabled.
	 *
	 * Some of the following lines of the algorithm EPCH are implemented here:
	 * - Lines 7-9: False Alarms handling at buffer W level (Case 2)
	 * - Line 10: if (size(W) 𝝐 (𝟏, 𝝁)) ->In warning window
	 * - Line 11: Train the background classifier
	 * - Line 12: Add instance to the buffer of instances during warning
	 * - Line 13-15: Update centroids / prototypes.
	 * - Line 16-20: If a warning is detected, start warning window and clear buffer W.
	 *
	 * The steps followed for this can be seen below:
	 * - Step 1 Update warning detection adding latest error  /*********** warning detection ***********
	 * - Step 2 Check for False Alarm (Case 2) - Lines 7-10
	 * - Step 3 If the classifier is in the warning window, train the bkg classifier and add the current instance to W.
	 * - Step 3.1 Otherwise update the topology (the topology does not update during warning)
	 * - Step 4 Check if there was a change (warning signal). If so, start warning window;
	 * 		 	In case of false alarm this triggers warning again and the bkg learner gets replaced.
	 * - Step 4.1 Update the warning detection object for the current object.
	 * 			  This effectively resets changes made to the object while it was still a bkglearner.
	 * - Step 4.2 Start warning window.
	 *
	 */
	protected void detectWarning(Instance instance, boolean correctlyClassified) {
		// Step 1: Update the WARNING detection method
		this.warningDetectionMethod.input(correctlyClassified ? 0 : 1);
		
		// Step 2: Check for False Alarm case 2 (Lines 7-9)
		if (this.W.size() >= this.warningWindowSizeThresholdOption.getValue()) resetWarningWindow(); // Line 8
		 		
		// Step 3: Either warning window training/buffering or topology update (Lines 10-15)
		if (this.W.size() >= 1 && this.W.size() < this.warningWindowSizeThresholdOption.getValue()) { // &&
				// this.ensemble.bkgLearner != null) { // when W is in that range, bkgLearner != null (tested), so the condition is not required
				this.active.bkgLearner.classifier.trainOnInstance(instance); // Line 11
			this.W.add(instance); // Line 12
		} else this.topology.trainOnInstanceImpl(instance); // Step 3.1: Lines 13-15 (TODO: should we feed a given instance to GNG many times?)

		// Step 4: line 16: warning detected?
		if (this.warningDetectionMethod.getChange()) {
            resetWarningWindow(); // Step 4.1 (Line 19)
			startWarningWindow(); //Step 4.2
		}
	}
	
	protected void resetWarningWindow(){
		this.active.bkgLearner = null; // Lines 8 and 19
		this.active.internalWindowEvaluator = null;
		this.active.tmpCopyOfClassifier = null;
		this.warningDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.warningDetectionMethodOption)).copy(); // restart warning
		this.W.delete(); // Lines 8 and 19 (it also initializes the object W)
		this.CH.decreaseNumberOfWarnings(0); // update applicable concepts
	}
	
	/**
	 * This starts the warning window event
	 *
	 * The next steps are followed:
	 * - 1 Update last error and make a backup of the current classifier in a concept object
	 * 		(the active one will be in use until the Drift is confirmed).
	 * - 2 Update of objects with warning.
	 * - 3 If the concept internal evaluator has been initialized for any other classifier on warning,
	 * 		add window size and last error of current classifier on warning.
	 *		Otherwise, initialize a new internal evaluator for the concept
	 * */
	protected void startWarningWindow() {  // TODO: review this as I've moved things.
		this.lastWarningOn = this.instancesSeen;
		this.numberOfWarningsDetected++;

		// Step 1 Update last error and make a backup of the current classifier
		if (!this.disableRecurringDriftDetectionOption.isSet()) {
			this.active.saveCurrentConcept(this.instancesSeen); // line 17: Save a tmp copy of c as snapshot
		}
		// Step 2: Update of objects with warning.
		if (!disableRecurringDriftDetectionOption.isSet()) {
			// this.ensemble.internalWindowEvaluator = null; (replaced anyway by the next line) (TODO: why would this be in RCARF?)
			this.active.createInternalEvaluator();
			this.CH.increaseNumberOfWarnings(0, this.active, this.lastError); // 0 is the active learner pos in an ensemble
		}
		if (this.eventsLogFile != null && this.logLevelOption.getValue() >= 1) logEvent(getWarningEvent()); // Log this

		// Step 3: Create background Classifier
		this.active.createBkgClassifier(this.lastWarningOn); // line 18: create background classifier
	}


	/**
	 * This method selects the next concept classifier and closest group topology when a drift is raised.
	 * 	Pselected is: a new P (Pn) in case of bkgDrift; Pc in case of false alarm; and Ph in case of recurring drift.
	 *
	 * The next steps are followed:
	 * - 0 Set false in case of drift at false as default.
	 *     Included for cases where driftDecisionMechanism > 0 and recurring drifts are enabled.
	 * - 1 Compare DT results using Window method and pick the best one between CH and bkg classifier.
	 *     It returns the best classifier in the object of the bkgLearner if there is not another base classifier
	 *      with lower error than active classifier (and driftDecisionMechanism > 0), then a false alarm is raised.
	 *     This step belong to line 32 in the algorithm: c = FindClassifier(c, b, GH) -> Assign best transition to next state.
	 * - 2 Orchestrate all the actions if the drift is confirmed and there is not a false alarm.
	 * - 3 Decrease amount of warnings in concept history and from evaluators
	 * - 4 reset base learner
	 *
	 * Lines of the algorithm Lines 22-25 are implemented here:
	 * -----------
	 * Insertion in CH (Lines 24-28)
	 * 	line 24: get prototypes from topology
	 * 	line 25: Group for storing old state
	 * 	line 26-28: create a placeholder for a group represented by 'tmpPrototypes'
	 * Retrieval from CH and refresh (lines 29-35)
	 * 	line 29: push current classifier to Gc
	 * 	line 30: Update topology on Gc
	 * 	line 31: Group for retrieval of next state
	 *  lines 32-33: In method switchActiveClassifier
	 * 	line 34: Add the examples during warning to a new topology.
	 * 	line 35: Empty list W
	 */
	protected void detectDrift(boolean correctlyClassified) {
		/*********** drift detection ***********/
		// Update the DRIFT detection method
		this.driftDetectionMethod.input(correctlyClassified ? 0 : 1);
		// Check if there was a change
		if (this.driftDetectionMethod.getChange()) { // line 22-23 drift detected?
			this.lastDriftOn = this.instancesSeen;
			this.numberOfDriftsDetected++;
			boolean falseAlarm = false; // Set false alarms (case 1) at false as default
			
			// Retrieval from CH
			if (!this.disableRecurringDriftDetectionOption.isSet()) // step 1
				// Start retrieval from CH (TODO: W should only be used if warning detection is enabled. same for topologies?)
				falseAlarm = switchActiveClassifier(this.CH.findGroup(this.W)); // lines 31-33 of the algorithm
			else if (this.eventsLogFile != null && this.logLevelOption.getValue() >= 1)
				logEvent(getBkgDriftEvent());  // TODO. Refactor this logEvent function so always
											  //		   it's inside of 'registerDrift' and not wrapping it
			if (!falseAlarm) {  // Step 2
				// Insertion in CH (Lines 24-27)
				Instances tmpPrototypes = this.topology.getPrototypes(); // line 24
				int previousGroup = this.CH.findGroup(tmpPrototypes); // line 25
				if (this.CH.size() == 0 || previousGroup == -1) { // line 26
					previousGroup = this.groupId++;
					this.CH.createNewGroup(previousGroup, tmpPrototypes, this.newTopology); // line 27
				} pushToConceptHistory(previousGroup); // lines 29-30
				
				if (!this.disableRecurringDriftDetectionOption.isSet())
					this.CH.decreaseNumberOfWarnings(0); //, previousGroup); // step 3
				this.active.reset(); // reset base classifier  (step 4)
				if (this.newTopology != null) this.topology = updateExtraTopology(this.newTopology, this.W); // line 34
				this.W.delete(); // line 35
			}
		}
	}

	protected void pushToConceptHistory(int previousGroup) {
		// Move copy of active classifier made before warning to Concept History.
		this.active.tmpCopyOfClassifier.addHistoryID(this.CHid++); // ConceptHistory.nextID())
		this.CH.addLearnerToGroup(previousGroup, this.active.tmpCopyOfClassifier); // line 29
		this.CH.setGroupTopology(previousGroup, mergeTopologies(this.CH.getTopologyFromGroup(previousGroup))); // line 30
	}

	// DRIFT ACTIONS

	/***
	 * Register false alarm an update variables consequently
	 * (both the active classifier and the ensemble topology will remain being the same)
	 */
	protected boolean registerDriftFalseAlarm() {
		if (this.eventsLogFile != null && this.logLevel >= 0) logEvent(getFalseAlarmEvent());
		this.newTopology = null; // then Pn = Pc  (line 33)
		return true;
	}

	/***
	 * Register recurring drift an update variables consequently
	 * Copy the best recurring learner in the history group passed, and the topology of this group.
	 */
	protected void registerRecurringDrift(Integer indexOfBestRanked, int historyGroup) {
		if (this.eventsLogFile != null && this.logLevel >= 0)
			logEvent(getRecurringDriftEvent(indexOfBestRanked, historyGroup));
		this.active.bkgLearner = this.CH.copyConcept(historyGroup, indexOfBestRanked);
		this.newTopology = this.CH.getTopologyFromGroup(historyGroup);  // then Pn = Ph  (line 33)
	}

	/***
	 * Register background drift an update variables consequently
	 * Pselected is a new P in case of background drift
	 */
	protected void registerBkgDrift() {
		// Register background drift
		if (this.eventsLogFile != null && this.logLevel >= 0)
			logEvent(getBkgDriftEvent());
		this.newTopology = new Topology(this.topologyLambdaOption, this.alfaOption, this.maxAgeOption,
				this.constantOption, this.BepsilonOption, this.NepsilonOption, this.classNotAsAnAttributeInTopologyOption); // line 33
		this.newTopology.resetLearningImpl();
	}

	// IDENTIFICATION OF NEXT STATE

	/**
	 * This method ranks all applicable base classifiers in the Concept History (CH)
	 * It also selects the next classifier to be active, or it raises a false alarm
	 * if the drift should be reconsidered.
	 *
	 * It implements the lines 32-33 of the algorithm.
	 *  lines 32-33: get topology of the group (methods 'registerDrift...') and retrieve the best classifier
	 *
	 * -----------------------------------------------------------------------------------------------------------------
	 * False alarms depend on the drift decision mechanism
	 * -----------------------------------------------------------------------------------------------------------------
	 *
	 * When driftDecisionMechanism == 0, if bkgLearner == null, false alarms cannot
	 * be raised. A comparison against CH is not possible as there is no bkg learner
	 * trained. In this case, a drift signal has been raised and it cannot be
	 * stopped without false alarms. A bkg drift applies as only option available.
	 *
	 * When drift decision mechanism == 1 or 2, then false alarms are taken into
	 * consideration for drifts (the warning will be still active even if a false
	 * alarm is raised for a drift in the same active classifier). If the background
	 * learner is NULL, we consider that the drift signal may have been caused by a
	 * too sensitive drift detection parameterization. In this case, it's clearly
	 * too soon to change the active classifier. Therefore we raise a drift signal.
	 *
	 * When drift decision mechanism == 2, we also raise a false alarm when the
	 * active classifier obtains less error than the bkg classifier and all of the
	 * classifiers from the CH.
	 *
	 * -----------------------------------------------------------------------------------------------------------------
	 * If the active classifier is not the best available choice / false alarm is
	 * raised, the following logic applies:
	 * -----------------------------------------------------------------------------------------------------------------
	 * If bkgBetterThanCHbaseClassifier == False, the minimum error of the base
	 * classifiers in the CH is not lower than the error of the bkg classifier.
	 * Then, register background drift.
	 *
	 * If CHranking.size() == 0, no applicable concepts for the active classifier in
	 * the concept history. Then, we register background drift. Otherwise, a
	 * recurring drift is the best option.
	 *
	 * @param historyGroup
	 *
	 */
	protected boolean switchActiveClassifier(int historyGroup) {
	    int indexOfBestRanked = -1;
		double errorOfBestRanked = -1.0;
		HashMap<Integer, Double> ranking = new HashMap<Integer, Double> ();
		
		// 1 Raise a false alarm for the drift if the background learner is not ready (Case 1)
		if (this.driftDecisionMechanismOption.getValue() > 0 && this.active.bkgLearner == null)
			return registerDriftFalseAlarm();
		    
		// 2 Retrieve best applicable classifier from Concept History (if a CH group applies)
		if (historyGroup != -1) ranking = rankConceptHistoryClassifiers(historyGroup);
		if (ranking.size() > 0) {
			indexOfBestRanked = getMinKey(ranking); // find index of concept with lowest value (error)
			errorOfBestRanked = Collections.min(ranking.values());
		}
		// 3 Compare this against the background classifier and make the decision.
		if (this.driftDecisionMechanismOption.getValue() == 2) {
			if (activeBetterThanBKGbaseClassifier()) {
				if (ranking.size() > 0 && !activeBetterThanCHbaseClassifier(errorOfBestRanked))
					registerRecurringDrift(indexOfBestRanked, historyGroup);
				// False alarm if active classifier is still the best one and when there are no applicable concepts.
				else return registerDriftFalseAlarm();
			} else {
				if (ranking.size() > 0 && bkgBetterThanCHbaseClassifier(errorOfBestRanked))
					registerRecurringDrift(indexOfBestRanked, historyGroup);
				else registerBkgDrift();
			}
		// Drift decision mechanism == 0 or 1 (in an edge case where the bkgclassifier is still NULL, we ignore the comparisons) (Case 1)
		} else {
			if (ranking.size() > 0 && this.active.bkgLearner != null
					&& bkgBetterThanCHbaseClassifier(errorOfBestRanked))
				registerRecurringDrift(indexOfBestRanked, historyGroup);
			else
				registerBkgDrift();
			
		} return false; // No false alarms raised at this point

	}

	protected Topology mergeTopologies(Topology CHtop) {		

		// way 2: old prototypes may be more important here due to the way how GNG works
		return updateExtraTopology(CHtop, this.topology.getPrototypes());
	}

	/**
	 * This auxiliary function updates either old or new topologies that will be merged with, compared with, or will replace to the current one.
	 * */
	protected Topology updateExtraTopology(Topology top, Instances w2) {
		
		/*
		int trainType = 0;
		if(stopPercentageOption.getValue() > 0)
			top.stoppingCriteriaOption.setValue((int)((this.stopPercentageOption.getValue() * (double) w2.size()) / 100.0));
		
		if(trainType==0) {
			// We add them several times till achieving the stopping criteria as in iGNGSVM
			// The effect of this would be a GNG topology that may not be able to keep expanding, which may be undesired in EPCH.
			for (int i=0; top.getNumberOfPrototypesCreated()<top.stoppingCriteriaOption.getValue(); i++){
				top.trainOnInstanceImpl((Instance) w2.get(i));
		        	if(i+1==w2.numInstances()) i = -1;
		    }
			
		} else { */
			// we add them once
			for (int instPos = 0; instPos < w2.size(); instPos++) {
				top.trainOnInstanceImpl(w2.get(instPos));
			}
		// }
		return top; // if topology (Pc) is global, then we don´t need to return this here
	}
		
	/* Compute distances between Instances as seen in GNG for arrays (GUnit objects). **/
	public double dist(Instance w1,Instance w2){
		double sum = 0;
		for (int i = 0; i < w1.numAttributes(); i++) {
			sum += Math.pow(w1.value(i)-w2.value(i),2);
		}
		return Math.sqrt(sum);
	}
	
    /**
     * This function ranks the best concepts from a given group of the Concept History
     * -----------------------------------
	 * This only takes into consideration Concepts sent to the Concept History
	 * 	after the current classifier raised a warning (see this consideration in reset*)
	 *
	 * The Concept History owns only one learner per historic concept.
	 * 	But each learner has a different window size and error.
	 *
	 * this.indexOriginal - pos of this classifier with active warning in ensemble
	 */
	protected HashMap<Integer, Double> rankConceptHistoryClassifiers(int historyGroup) {
		HashMap<Integer, Double> CHranking = new HashMap<Integer, Double>();

		for (Concept auxConcept : this.CH.getConceptsFromGroup(historyGroup))
			if (auxConcept.ConceptLearner.internalWindowEvaluator != null
					&& auxConcept.ConceptLearner.internalWindowEvaluator.containsIndex(0)) { // 0 is the pos of the active learner in an ensemble
				CHranking.put(auxConcept.getHistoryIndex(),
						((DynamicWindowClassificationPerformanceEvaluator) auxConcept.ConceptLearner.internalWindowEvaluator)
								.getFractionIncorrectlyClassified(0));
			}
		return CHranking;
	}

	/**
	 * Aux method for getting the best classifier (used to rank concepts from a group in the CH)
	 * */
	protected Integer getMinKey(Map<Integer, Double> map) {
		Integer minKey = null;
		double minValue = Double.MAX_VALUE;
		for (Integer key : map.keySet()) {
			double value = map.get(key);
			if (value < minValue) {
				minValue = value;
				minKey = key;
			}
		} return minKey;
	}

	protected boolean activeBetterThanBKGbaseClassifier() {
		// If drift decision mechanism is == 2
		return (((DynamicWindowClassificationPerformanceEvaluator) this.active.internalWindowEvaluator)
			.getFractionIncorrectlyClassified(this.active.indexOriginal) <= ((DynamicWindowClassificationPerformanceEvaluator)
			this.active.bkgLearner.internalWindowEvaluator).getFractionIncorrectlyClassified(this.active.bkgLearner.indexOriginal));
		// this.indexOriginal - pos of this classifier with active warning in ensemble
		// return ((this.ensemble.evaluator.getFractionIncorrectlyClassified() <= ((DynamicWindowClassificationPerformanceEvaluator)
		//		this.ensemble.bkgLearner.internalWindowEvaluator).getFractionIncorrectlyClassified(this.ensemble.bkgLearner.indexOriginal)));
	}

	protected boolean activeBetterThanCHbaseClassifier(double bestFromCH) {
		// If drift decision mechanism is == 2
		return (((DynamicWindowClassificationPerformanceEvaluator) this.active.internalWindowEvaluator)
			.getFractionIncorrectlyClassified(this.active.indexOriginal) <= bestFromCH);

		// this.indexOriginal - pos of this classifier with active warning in ensemble
		// return (this.ensemble.evaluator.getFractionIncorrectlyClassified() <= bestFromCH); (old comparison)
	}

	protected boolean bkgBetterThanCHbaseClassifier(double bestFromCH) {
		// this.bkgLearner.indexOriginal - pos of bkg classifier if it becomes active in the ensemble (always same pos than the active)
		return (bestFromCH <= ((DynamicWindowClassificationPerformanceEvaluator) this.active.bkgLearner.internalWindowEvaluator)
				.getFractionIncorrectlyClassified(this.active.bkgLearner.indexOriginal));

	}
	
	///////////////////////////////////////
	//
	// LOGGING FUNCTIONS
	// -----------------------------------

	public Event getTrainExampleEvent() {
		String[] eventLog = { String.valueOf(instancesSeen), "Train example", String.valueOf(0),
				String.valueOf(this.active.evaluator.getPerformanceMeasurements()[1].getValue()),
				this.warningDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				String.valueOf(this.instancesSeen), String.valueOf(this.active.evaluator.getFractionIncorrectlyClassified()),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings().size(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getNumberOfActiveWarnings(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings(): "N/A"),
				"N/A", "N/A", "N/A" };

		return (new Event(eventLog));
	}

	public Event getWarningEvent() {

		// System.out.println();
		System.out.println("-------------------------------------------------");
		System.out.println("WARNING ON IN MODEL #"+0+". Warning flag status (activeClassifierPos, Flag): "+CH.getWarnings());
		System.out.println("CONCEPT HISTORY STATE AND APPLICABLE FROM THIS WARNING  IS: "+CH.keySet().toString());
		System.out.println("-------------------------------------------------");
		// System.out.println();

		String[] warningLog = { String.valueOf(this.lastWarningOn), "WARNING-START", // event
				String.valueOf(0),
				String.valueOf(this.active.evaluator.getPerformanceMeasurements()[1].getValue()),
				this.warningDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				String.valueOf(this.instancesSeen), String.valueOf(this.active.evaluator.getFractionIncorrectlyClassified()),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings().size(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getNumberOfActiveWarnings(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings(): "N/A"),
				!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.keySet().toString(): "N/A",
				"N/A", "N/A" };
		// 1279,1,WARNING-START,0.74,{F,T,F;F;F;F},...

		return (new Event(warningLog));
	}

	public Event getBkgDriftEvent() {
		System.out.println("DRIFT RESET IN MODEL #"+0+" TO NEW BKG MODEL #"+this.active.bkgLearner.indexOriginal);
		String[] eventLog = {
				String.valueOf(this.lastDriftOn), "DRIFT TO BKG MODEL", String.valueOf(0),
				String.valueOf(this.active.evaluator.getPerformanceMeasurements()[1].getValue()),
				this.warningDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				String.valueOf(this.instancesSeen), String.valueOf(this.active.evaluator.getFractionIncorrectlyClassified()),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings().size(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getNumberOfActiveWarnings(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings(): "N/A"),
				"N/A", "N/A", "N/A" };
		return (new Event(eventLog));
	}

	public Event getRecurringDriftEvent(Integer indexOfBestRankedInCH, int group) {
		System.out.println(indexOfBestRankedInCH); // TODO: debugging
		System.out.println("RECURRING DRIFT RESET IN POSITION #"+0+" TO MODEL #"+
		CH.get(indexOfBestRankedInCH).groupList.get(indexOfBestRankedInCH).ensembleIndex);
		// +this.bkgLearner.indexOriginal);
		String[] eventLog = { String.valueOf(this.lastDriftOn), "RECURRING DRIFT", String.valueOf(0),
				String.valueOf(this.active.evaluator.getPerformanceMeasurements()[1].getValue()),
				this.warningDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				String.valueOf(this.instancesSeen), String.valueOf(this.active.evaluator.getFractionIncorrectlyClassified()),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings().size(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getNumberOfActiveWarnings(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings(): "N/A"),
				"N/A",
				String.valueOf(this.CH.get(group).groupList.get(indexOfBestRankedInCH).ensembleIndex),
				String.valueOf(this.CH.get(group).groupList.get(indexOfBestRankedInCH).createdOn) };
		return (new Event(eventLog));
	}

	public Event getFalseAlarmEvent() {
		System.out.println("FALSE ALARM IN MODEL #"+0);
		String[] eventLog = { String.valueOf(this.lastDriftOn), "FALSE ALARM ON DRIFT SIGNAL",
				String.valueOf(0),
				String.valueOf(this.active.evaluator.getPerformanceMeasurements()[1].getValue()),
				this.warningDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				String.valueOf(this.instancesSeen), String.valueOf(this.active.evaluator.getFractionIncorrectlyClassified()),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings().size(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getNumberOfActiveWarnings(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings(): "N/A"),
				"N/A", "N/A", "N/A" };
		return (new Event(eventLog));
	}

	// General auxiliar methods for logging events
	public Event getEventHeaders() {
		String[] headers = { "instance_number", "event_type", "affected_position", // former 'classifier'
				"voting_weight", // voting weight for the three that presents an event.
				"warning_setting", "drift_setting", "affected_classifier_created_on", "error_percentage",
				"amount_of_classifiers", "amount_of_active_warnings", "classifiers_on_warning", "applicable_concepts",
				"recurring_drift_to_history_id", "recurring_drift_to_classifier_created_on" };
		return (new Event(headers));
	}

	/**
	 * Method to register events such as Warning and Drifts in the event log file.
	 */
	public void logEvent(Event eventDetails) {
		// Log processed instances, warnings and drifts in file of events
		// # instance, event, affected_position, affected_classifier_id last-error, #classifiers;#active_warnings; classifiers_on_warning,
		// 		applicable_concepts_from_here, recurring_drift_to_history_id, drift_to_classifier_created_on
		this.eventsLogFile.println(String.join(";", eventDetails.getInstanceNumber(), eventDetails.getEvent(),
				eventDetails.getAffectedPosition(), eventDetails.getVotingWeigth(), // of the affected position
				eventDetails.getWarningSetting(), // WARNING SETTING of the affected position.
				eventDetails.getDriftSetting(), // DRIFT SETTING of the affected position.
				eventDetails.getCreatedOn(), // new, affected_classifier_was_created_on
				eventDetails.getLastError(),
				eventDetails.getNumberOfClassifiers(),
				eventDetails.getNumberOfActiveWarnings(), // #active_warnings
				eventDetails.getClassifiersOnWarning(),  // toString of list of classifiers in warning
				eventDetails.getListOfApplicableConcepts(), // applicable_concepts_from_here
				eventDetails.getRecurringDriftToClassifierID(), // recurring_drift_to_history_id
				eventDetails.getDriftToClassifierCreatedOn()));
		this.eventsLogFile.flush();
	}

	
	///////////////////////////////////////
	//
	// AUX CLASSES
	// -----------------------------------
	
	/**
	 * Inner class that represents a single tree member of the ensemble. It contains
	 * some analysis information, such as the numberOfDriftsDetected,
	 */
	protected final class EPCHBaseLearner {
		public int indexOriginal;
		public long createdOn;
		public Classifier classifier;
		public boolean isBackgroundLearner;
		public boolean isOldLearner; // only for reference

		// public boolean useBkgLearner; // (now always true in EPCH)
		// these flags are still necessary at this level
		public boolean useDriftDetector;
		public boolean useRecurringLearner;

		// Bkg learner
		protected EPCHBaseLearner bkgLearner;

		// Copy of main classifier at the beginning of the warning window for its copy in the Concept History
		protected Concept tmpCopyOfClassifier;

		// Statistics
		public BasicClassificationPerformanceEvaluator evaluator;

		// Internal statistics
		public DynamicWindowClassificationPerformanceEvaluator internalWindowEvaluator; // for bkg and CH classifiers
		protected double lastError;
		protected Window windowProperties;

		int warningWindowSizeThreshold = -1;

		private void init(int indexOriginal, Classifier classifier,
				BasicClassificationPerformanceEvaluator evaluatorInstantiated, long instancesSeen, // boolean useBkgLearner,
				boolean useDriftDetector, boolean isBackgroundLearner, boolean useRecurringLearner, boolean isOldLearner,
				Window windowProperties, DynamicWindowClassificationPerformanceEvaluator internalEvaluator, int warningWindowSizeThreshold) {

			this.indexOriginal = indexOriginal;
			this.createdOn = instancesSeen;

			this.classifier = classifier;
			this.evaluator = evaluatorInstantiated;
			
			this.useDriftDetector = useDriftDetector;
			this.warningWindowSizeThreshold = warningWindowSizeThreshold;
			
			this.isBackgroundLearner = isBackgroundLearner;
			// this.useBkgLearner = useBkgLearner;  // TODO: remove. now always true in EPCH

			this.useRecurringLearner = useRecurringLearner;
			if (useRecurringLearner) {
				this.windowProperties = windowProperties; // Window params
				this.isOldLearner = isOldLearner; // Recurring drifts
				this.internalWindowEvaluator = internalEvaluator; // only used in bkg and retrieved old classifiers
			}

		}

		public boolean correctlyClassifies(Instance instance) {
			return this.classifier.correctlyClassifies(instance);
		}

		public EPCHBaseLearner(int indexOriginal, Classifier classifier, BasicClassificationPerformanceEvaluator evaluatorInstantiated,
				long instancesSeen, // boolean useBkgLearner,
				boolean useDriftDetector, boolean isBackgroundLearner,
				boolean useRecurringLearner, boolean isOldLearner, Window windowProperties,
				DynamicWindowClassificationPerformanceEvaluator bkgInternalEvaluator, int warningWindowSizeThreshold) {
			init(indexOriginal, classifier, evaluatorInstantiated, instancesSeen, // useBkgLearner,
					useDriftDetector, isBackgroundLearner, useRecurringLearner, isOldLearner, windowProperties, bkgInternalEvaluator, warningWindowSizeThreshold);
		}

        /**
         *  This function resets the EPCH base classifiers
		 *
		 *  The steps followed in this method can be seen below:
		 *
		 * - Step 2.1 Update window size in window properties depending on window size inheritance flag (entry parameter/Option)
		 * - Step 2.3 Move copy of active classifier made before warning to Concept History and reset.
		 *			  Its history ID will be the last one in the history (= size)
		 */
		public void reset() {
			System.out.println("-------------------------------------------------");
			System.out.println("RESET (WARNING OFF) IN MODEL #"+this.indexOriginal+
					". Warning flag status (activeClassifierPos, Flag): "+CH.getNumberOfActiveWarnings());
			// System.out.println("-------------------------------------------------");
			
			// Transition to the best bkg or retrieved old learner
			if (//this.useBkgLearner &&
				this.bkgLearner != null) {
				if (this.useRecurringLearner) {
					this.tmpCopyOfClassifier = null; // reset tc.

					// 2.1 Update the internal evaluator properties
					this.bkgLearner.windowProperties.setSize(((this.bkgLearner.windowProperties.rememberWindowSize)
							? this.bkgLearner.internalWindowEvaluator.getWindowSize(this.bkgLearner.indexOriginal)
							: this.bkgLearner.windowProperties.windowDefaultSize));

					// 2.2 Inherit window properties / clear internal evaluator
					this.windowProperties = this.bkgLearner.windowProperties;
				}
				// 2.3 New active classifier is the best retrieved old classifier / clear background learner
				this.classifier = this.bkgLearner.classifier;
				this.evaluator = this.bkgLearner.evaluator;
				this.createdOn = this.bkgLearner.createdOn;
				this.bkgLearner = null;
				this.internalWindowEvaluator = null;

			} else {
				this.classifier.resetLearning();
				this.createdOn = instancesSeen;
			}
			this.evaluator.reset();
		}

		public void trainOnInstance(Instance instance, long instancesSeen) { // Line 5: (x,y) ← next(S)
			this.classifier.trainOnInstance(instance); // Line 6: ClassifierTrain(c, x, y) -> Train c on the current
														// instance (x, y).
		}

		public double[] getVotesForInstance(Instance instance) {
			DoubleVector vote = new DoubleVector(this.classifier.getVotesForInstance(instance));
			return vote.getArrayRef();
		}
                                                                                                                                  		
		public void createInternalEvaluator() {
			// Add also an internal evaluator (window) in the bkgEvaluator
			this.internalWindowEvaluator = new DynamicWindowClassificationPerformanceEvaluator(
				this.windowProperties.getSize(), this.windowProperties.getIncrements(),
				this.windowProperties.getMinSize(), this.lastError,
				this.windowProperties.getDecisionThreshold(), true, this.windowProperties.getResizingPolicy(),
				this.indexOriginal, "created for active classifier in ensembleIndex #" + this.indexOriginal);
			this.internalWindowEvaluator.reset();
		}
		
		 /**
		 * This method creates BKG Classifier in warning window
		 * The next steps are followed:
		 *  Step 1 Create a new bkgTree classifier
		 *  Step 2 Resets the evaluator
		 *  Step 3 Create a new bkgLearner object
		 * */
		public void createBkgClassifier(long lastWarningOn) {
		    
			// 1 Create a new bkgTree classifier
			Classifier bkgClassifier = this.classifier.copy();
			bkgClassifier.resetLearning();

			// 2 Resets the evaluator
			BasicClassificationPerformanceEvaluator bkgEvaluator = (BasicClassificationPerformanceEvaluator) this.evaluator
					.copy();
			bkgEvaluator.reset();
			System.out.println("------------------------------");
			System.out.println("Create estimator for BKG classifier in position: "+this.indexOriginal);
			
			// Add also an internal evaluator (window) in the bkgEvaluator
			DynamicWindowClassificationPerformanceEvaluator bkgInternalWindowEvaluator = null;
			if (this.useRecurringLearner) {
				bkgInternalWindowEvaluator = new DynamicWindowClassificationPerformanceEvaluator(
						this.windowProperties.getSize(), this.windowProperties.getIncrements(),
						this.windowProperties.getMinSize(), this.lastError,
						this.windowProperties.getDecisionThreshold(), true, this.windowProperties.getResizingPolicy(),
						this.indexOriginal, "created for BKG classifier in ensembleIndex #" + this.indexOriginal);
				bkgInternalWindowEvaluator.reset();
			} System.out.println("------------------------------");

			// 4 Create a new bkgLearner object
			this.bkgLearner = new EPCHBaseLearner(indexOriginal, bkgClassifier, bkgEvaluator, lastWarningOn, //this.useBkgLearner,
					this.useDriftDetector, true, this.useRecurringLearner, false,
					this.windowProperties, bkgInternalWindowEvaluator, this.warningWindowSizeThreshold);
		}

		/**
		* This method saves a backup of the active classifier that raised a warning
		* to be stored in the Concept History in case of drift.
		*
		* The next steps are followed:
	    	*  Step 1 Update last error before warning of the active classifier. This error is the total fraction of examples
	    	* 		incorrectly classified this classifier was active until now.]
		*  Step 2 Copy Base learner for Concept History in case of Drift and store it temporal object.
		*	 First, the internal evaluator will be null. It doesn't get initialized till once in the Concept History
		*		and the first warning arises. See it in startWarningWindow
		*  Step 3 Add the classifier accumulated error (from the start of the classifier) from the iteration before the warning
		*  A simple concept to be stored in the concept history that doesn't have a running learner.
		*  This doesn't train. It keeps the classifier as it was at the beginning of the training window to be stored in case of drift.
		*/
		public void saveCurrentConcept(long instancesSeen) {
			this.lastError = this.evaluator.getFractionIncorrectlyClassified(); // step 1

			EPCHBaseLearner tmpConcept = new EPCHBaseLearner(this.indexOriginal, this.classifier.copy(), // step 2
					(BasicClassificationPerformanceEvaluator) this.evaluator.copy(), this.createdOn, // this.useBkgLearner,
					this.useDriftDetector, true, useRecurringLearner, true, this.windowProperties.copy(), null,
					this.warningWindowSizeThreshold);
			this.tmpCopyOfClassifier = new Concept(tmpConcept, this.createdOn,
					this.evaluator.getPerformanceMeasurements()[0].getValue(), instancesSeen);

			this.tmpCopyOfClassifier.setErrorBeforeWarning(this.lastError); // step 3
		}
	}

	protected class Concept {
		protected int ensembleIndex; // position that it had in the ensemble. for reference only.
		protected int historyIndex; // id in concept history

		// Stats
		protected long createdOn;
		protected long instancesSeen;
		protected double classifiedInstances;
		protected double errorBeforeWarning;

		// Learner
		public EPCHBaseLearner ConceptLearner;

		// Constructor
		public Concept(EPCHBaseLearner ConceptLearner, long createdOn, double classifiedInstances, long instancesSeen) {
			// Extra info
			this.createdOn = createdOn;
			this.instancesSeen = instancesSeen;
			this.classifiedInstances = classifiedInstances;
			this.ensembleIndex = ConceptLearner.indexOriginal;

			// Learner
			this.ConceptLearner = ConceptLearner;
		}

		public void addHistoryID(int id) {
			this.historyIndex = id;
		}

		public int getEnsembleIndex() {
			return this.ensembleIndex;
		}

		public int getHistoryIndex() {
			return this.historyIndex;
		}
		
		public DynamicWindowClassificationPerformanceEvaluator getInternalEvaluator() {
			return this.ConceptLearner.internalWindowEvaluator;
		}
		
		public void setInternalEvaluator(DynamicWindowClassificationPerformanceEvaluator intEval) {
			this.ConceptLearner.internalWindowEvaluator = intEval;
		}
		
		public void addModelToInternalEvaluator(int ensemblePos, double lastError, int windowSize) {
			this.ConceptLearner.internalWindowEvaluator.addModel(ensemblePos, lastError, windowSize);
		}
		
		public void deleteModelFromInternalEvaluator(int ensemblePos) {
			this.ConceptLearner.internalWindowEvaluator.deleteModel(ensemblePos);
		}
		
		/* The base learner ePos and its window ePW refer to the applicable concept that will be compared from this point to this concept. */
		public void newInternalWindow(EPCHBaseLearner ePos, Window ePW) {
			DynamicWindowClassificationPerformanceEvaluator tmpInternalWindow =
					new DynamicWindowClassificationPerformanceEvaluator(ePW.getSize(), ePW.getIncrements(),ePW.getMinSize(),
							ePos.lastError, ePW.getDecisionThreshold(), ePW.getDynamicWindowInOldClassifiersFlag(), ePW.getResizingPolicy(),
							ePos.indexOriginal, "created for old-retrieved classifier in ensembleIndex #" + ePos.indexOriginal);
			tmpInternalWindow.reset();
			this.setInternalEvaluator(tmpInternalWindow);
		}
		
		public int getAmountOfApplicableModels() {
			return this.ConceptLearner.internalWindowEvaluator.getAmountOfApplicableModels();
		}
		
		public void addResultToInternalEvaluator(InstanceExample inst, double[] voteRef) {
			this.ConceptLearner.internalWindowEvaluator.addResult(inst, voteRef);
		}

		public EPCHBaseLearner getBaseLearner() {
			return this.ConceptLearner;
		}

		// Setters
		public void setErrorBeforeWarning(double value) {
			this.errorBeforeWarning = value;
		}
	}
	
	/**
	 * TOPOLOGY RELATED METHODS AND META CLASS FOR CLUSTERING/TOPOLOGY SUMMARY ALGORITHM
	 * */
	protected class Topology {
		
		// TODO: options should come from the meta class for topology
		public IntOption GNGLambdaOption = new IntOption("lambda", 'l', "GNG Lambda", 100);
		public IntOption maxAgeOption = new IntOption("maxAge", 'm', "MaximumAge", 200);
		public FloatOption alfaOption = new FloatOption("alfa", 'a', "Alfa", 0.5);
		public FloatOption constantOption = new FloatOption("d", 'd', "d", 0.995);
		public FloatOption BepsilonOption = new FloatOption("epsilonB", 'Z', "EpsilonB", 0.2);
		public FloatOption NepsilonOption = new FloatOption("epsilonN", 'K', "EpsilonN", 0.006);
		public IntOption stoppingCriteriaOption = new IntOption("stoppingCriteria", 'c', "Stopping criteria", 0);
		public FlagOption classAsAttributeOption = new FlagOption("classAsFeature", 'b',
				"Should the class be considered as a feature in the topology?");

		protected GNG learner; // TODO: this should be a meta class so the algorithm can be changed
		private Instance auxInst;
		

		public Topology(IntOption GNGLambdaOption, FloatOption alfaOption, IntOption maxAgeOption, FloatOption constantOption,
				FloatOption BepsilonOption, FloatOption NepsilonOption, FlagOption classNotAsAttributeOption) {
			this.GNGLambdaOption = GNGLambdaOption;
			this.alfaOption = alfaOption;
			this.maxAgeOption = maxAgeOption;
			this.constantOption = constantOption;
			this.BepsilonOption = BepsilonOption;
			this.NepsilonOption = NepsilonOption;
			
			// inverting parameter as for topology objects the default flag value should be set
			this.classAsAttributeOption = classNotAsAttributeOption;
			this.classAsAttributeOption.setValue(!classNotAsAttributeOption.isSet());
			
			this.learner = new GNG(this.GNGLambdaOption, this.alfaOption, this.maxAgeOption, this.constantOption,
					this.BepsilonOption, this.NepsilonOption, this.classAsAttributeOption);
		}
		
		protected void resetLearningImpl() {
			this.learner.resetLearningImpl();
		}
		
		public void trainOnInstanceImpl(Instance inst) {
			if (this.auxInst == null) this.auxInst = inst.copy(); // Aux variable for conversions
			this.learner.trainOnInstanceImpl(inst);
		}
	
		protected ArrayList<double[]> prototypeToArray(ArrayList<GUnit> tmpPrototypes) {
			ArrayList <double[]> converted = new ArrayList <double[]> ();
			for(GUnit prototype: tmpPrototypes)
				converted.add(prototype.w);
			return converted;
		}
		
		protected int getNumberOfPrototypesCreated() {
			return this.learner.getNumberOfPrototypesCreated();
		}
		
		protected Instances prototypesToUnsupervisedInstances(ArrayList<GUnit> tmpPrototypes) {
			Instances tmp = new Instances ((this.auxInst).dataset());
			Instance inst = null;
			
			for(GUnit prototype: tmpPrototypes) {
				inst = (Instance) this.auxInst.copy();
				for (int j = 0; j < prototype.w.length; j++)
					inst.setValue(j, prototype.w[j]);
				// TODO: inst.setClassValue(labels.get(i)); (This should not be necessary as in first instance we would also feed the class as part of the topologies,
				// and we don´t need to train the ensemble with those prototypes as we do in iGNGSVM ).
				tmp.add(inst);
			}
			return tmp;
		}
		
		public Instances getPrototypes() {
			return prototypesToUnsupervisedInstances(this.learner.getS());
		}

	}
	
	protected class Group {
		int id;
		Topology topology;
		public ConcurrentHashMap<Integer, Concept> groupList; // List of concepts per group

		public Group(int id, Topology top) {
			this.id = id; // nextID();
			this.topology = top;
			this.groupList = new ConcurrentHashMap<Integer, Concept>();
		}

		public EPCHBaseLearner copyConcept(int key) {
			EPCHBaseLearner aux = groupList.get(key).getBaseLearner();
			return aux;
		}

		// Getters
		public int getID() {
			return id;
		}

		public Instances getTopologyPrototypes() {
			return topology.getPrototypes(); // these instances don't belong to a class (unsupervised)
		}
		
		public void setTopology(Topology topology2) {
			this.topology = topology2;
		}
		
		public Topology getTopology() {
			return this.topology;
		}
		
		public Collection<Concept> values(){
			return this.groupList.values();
		}
		
		public void put(int idx, Concept learner) {
			this.groupList.put(idx, learner);
		}

	}

	/***
	 * Static and concurrent for all DTs that run in parallel Concept_history = (list of concept_representations)
	 */
	protected class ConceptHistory {
		
		double maxDistanceThreshold;

		// Concurrent Concept History List
		protected HashMap<Integer, Group> history; // now this is a list of groups

		// List of ensembles with an active warning used as to determine if the history list evaluators should be in use
		protected HashMap<Integer, Boolean> classifiersOnWarning;
		
		public ConceptHistory() {
			this.history = new HashMap<Integer, Group>();
			this.classifiersOnWarning = new HashMap<Integer, Boolean>();
			this.maxDistanceThreshold = Float.MAX_VALUE;
		}
		
		public ConceptHistory(double distThreshold) {
			this.history = new HashMap<Integer, Group>();
			this.classifiersOnWarning = new HashMap<Integer, Boolean>();
			this.maxDistanceThreshold = distThreshold;
		}
		
		public EPCHBaseLearner copyConcept(int group, int key) {
			EPCHBaseLearner aux = history.get(group).copyConcept(key);
			return aux;
		}
		
		public int getNumberOfActiveWarnings() {
			int count = 0;
			for (Boolean value : classifiersOnWarning.values()) if (value) count++;
			return count;
		}
		
		/**
		 * When the concept is added the first time, it doesn't have applicable classifiers.
		 * They are not inserted until the first warning. So the Concept History only runs over warning windows.
		 */
		public void updateHistoryErrors(Instance inst) {
			for (int historyGroup : this.history.keySet()) {
				for (Concept learner : this.history.get(historyGroup).groupList.values()) {
					DoubleVector oldLearnerVote = new DoubleVector(learner.ConceptLearner.getVotesForInstance(inst));
					if (learner.getInternalEvaluator() != null && learner.getInternalEvaluator().getAmountOfApplicableModels() > 0)
						learner.addResultToInternalEvaluator(new InstanceExample(inst), oldLearnerVote.getArrayRef());
				}
			}
		}
		
		/**
		 * This updates the error of CH classifiers with warning.
		 * The next steps are followed:
		 * - 1 It turns on windows flag in the Concept History.
		 * 		Also, if the Concept History is ready and it contains old classifiers,
		 * 		it adds both prior estimation and window size to each concept history learner.
		 * - 2 If the concept internal evaluator has been initialized for any other classifier on warning,
		 * 		add window size and last error of current classifier on warning.
		 * - 3 Otherwise, initialize a new internal evaluator for the concept
		 *
		 * @parameter ePos: base classifier on a given ensemble position
		 * @parameter lastError of the above-mentioned base classifier
		 */
		public void increaseNumberOfWarnings(int ensemblePos, EPCHBaseLearner ePos, double lastError) {
			this.classifiersOnWarning.put(ePos.indexOriginal, true);
			
			Window ePW = ePos.windowProperties;
			if (history != null && history.size() > 0) {
				// This adds it as an applicable concept to all the groups, as we don't know its group yet
				for (int historyGroup : history.keySet()) { // TODO: see in method decreaseNumberOfWarnings
					for (Concept learner : history.get(historyGroup).groupList.values()) {
						if (learner.getInternalEvaluator() != null) { // Step 2
							System.out.println("ADDING VALUES TO INTERNAL EVALUATOR OF CONCEPT "+	learner.historyIndex+" IN POS "+ePos.indexOriginal);
							learner.addModelToInternalEvaluator(ensemblePos, lastError, ePW.windowSize);

						} else { // Step 3: Otherwise, initialize a new internal evaluator for the concept
							System.out.println("INSTANCIATING FOR THE FIRST TIME INTERNAL EVALUATOR FOR CONCEPT "+learner.historyIndex+" IN POS "+ePos.indexOriginal);
							learner.newInternalWindow(ePos, ePW);
						}
					}
				}
			}
		}
		
		/**
		 * This method decreases the amount of warnings in concept history and from evaluators
		 * */
		public void decreaseNumberOfWarnings(int ensemblePos) { // , int oldGroup) {
			this.classifiersOnWarning.put(ensemblePos, false);
			if (this.history != null && this.history.size() > 0) {
				// This adds it as an applicable concept to all the groups, as we don't know its group yet
				// TODO: There should be only internal evaluators between classifiers inside a given group?
				for (int historyGroup : history.keySet()) {
					for (Concept learner : this.history.get(historyGroup).values()) {
						// TODO: use oldGroup instead of historyGroup (then change it also when increase it warnings (performance improvement)
						if (learner.getInternalEvaluator() != null && learner.getInternalEvaluator().containsIndex(ensemblePos)) {
							learner.deleteModelFromInternalEvaluator(ensemblePos);
							if (learner.getAmountOfApplicableModels() == 0) learner.setInternalEvaluator(null);  // TODO: implement this in RCARF / Evolving RCARF once tested
						}
					}
				}
			}
		}
		
		/** Creation of new group and pushing of this to the CH */
		protected void createNewGroup(int groupId, Instances tmpPrototypes, Topology newTopology) {
			Group g = new Group(groupId, newTopology);
			this.history.put(groupId, g); // the id is there twice to keep track of it and for testing purposes.
		}
		
		/**
		 * This method receives the current list of training examples received during
		 * the warning window and checks what's the closest group.
		 */
		protected int findGroup(Instances w2) {
			double min = Double.MAX_VALUE;
			double dist = 0;
			int group = -1;

			for (Group g : this.history.values()) {
				dist = getAbsoluteSumDistances(w2, g.getTopologyPrototypes());
				if (dist < min) {
					min = dist;
					group = g.getID();
				}
			}
			if (dist < this.maxDistanceThreshold) {
				return group;
			} else
				return -1;
		}
		
		/**
		 * This function computes the sum of the distances between every prototype in
		 * the topology passed and the current list of instances during warning (W)
		 */
		protected double getAbsoluteSumDistances(Instances w1, Instances w2) {
			double totalDist = 0.0;

			for (int instPos1 = 0; instPos1 < w1.size(); instPos1++) {
				for (int instPos2 = 0; instPos2 < w2.size(); instPos2++) {
					totalDist += dist(w1.get(instPos1), w2.get(instPos2));
				}
			} return totalDist;
		}

		
		public int size() {
			return this.history.size();
		}
		
		public Set<Integer> keySet() {
			return this.history.keySet();
		}
		
		private Group get(int key) {
			return this.history.get(key);
		}
		
		public HashMap<Integer, Boolean> getWarnings() {
			return this.classifiersOnWarning;
		}
		
		public Topology getTopologyFromGroup(int key) {
			return this.get(key).getTopology();
		}
				
		public void setGroupTopology(int groupID, Topology top) {
			this.history.get(groupID).setTopology(top); // line 30
		}
		
		public void addLearnerToGroup(int groupID, Concept learner) {
			this.history.get(groupID).put(learner.historyIndex, learner); // line 29
			// TODO: test that it changes effectively by updating it this way
		}
		
		public Collection<Concept> getConceptsFromGroup(int groupID){
			return this.get(groupID).values();
		}
	}

	/** Window-related parameters for classifier internal comparisons during the warning window */
	protected class Window {
		int windowSize;
		int windowDefaultSize;
		int windowIncrements;
		int minWindowSize;
		int windowResizePolicy;
		double decisionThreshold;
		boolean rememberWindowSize;
		boolean backgroundDynamicWindowsFlag;

		public Window(int windowSize, int windowIncrements, int minWindowSize, double decisionThreshold,
				boolean rememberWindowSize, boolean backgroundDynamicWindowsFlag, int windowResizePolicy) {
			this.windowSize = windowSize;
			// the default size of a window could change overtime if there is window size inheritance enabled
			this.windowDefaultSize = windowSize;
			this.windowIncrements = windowIncrements;
			this.minWindowSize = minWindowSize;
			this.decisionThreshold = decisionThreshold;
			this.backgroundDynamicWindowsFlag = backgroundDynamicWindowsFlag;
			this.windowResizePolicy = windowResizePolicy;
			this.rememberWindowSize = rememberWindowSize;
		}

		public Window copy() {
			return new Window(this.windowSize, this.windowIncrements, this.minWindowSize, this.decisionThreshold,
					this.rememberWindowSize, this.backgroundDynamicWindowsFlag, this.windowResizePolicy);
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

		public boolean getRememberSizeFlag() {
			return this.rememberWindowSize;
		}

		public void setRememberSizeFlag(boolean flag) {
			this.rememberWindowSize = flag;
		}

		public int getResizingPolicy() {
			return this.windowResizePolicy;
		}

		public void setResizingPolicy(int value) {
			this.windowResizePolicy = value;
		}

		public boolean getDynamicWindowInOldClassifiersFlag() {
			return this.backgroundDynamicWindowsFlag;
		}

		public void getDynamicWindowInOldClassifiersFlag(boolean flag) {
			this.backgroundDynamicWindowsFlag = flag;
		}

	}

	/** Object for events so the code is cleaner */
	public class Event {
		String instanceNumber;
		String event;
		String affectedPosition;
		String votingWeigth;
		String warningSetting;
		String driftSetting;
		String createdOn;
		String lastError;
		String numberOfClassifiers;
		String numberOfActiveWarnings;
		String classifiersOnWarning;
		String listOfApplicableConcepts;
		String recurringDriftToClassifierID;
		String driftToClassifierCreatedOn;

		// Constructor from array
		public Event(String[] eventDetails) {
			instanceNumber = eventDetails[0];
			event = eventDetails[1];
			affectedPosition = eventDetails[2];
			votingWeigth = eventDetails[3];
			warningSetting = eventDetails[4];
			driftSetting = eventDetails[5];
			createdOn = eventDetails[6];
			lastError = eventDetails[7];
			numberOfClassifiers = eventDetails[8];
			numberOfActiveWarnings = eventDetails[9];
			classifiersOnWarning = eventDetails[10];
			listOfApplicableConcepts = eventDetails[11];
			recurringDriftToClassifierID = eventDetails[12];
			driftToClassifierCreatedOn = eventDetails[13];
		}

		// Getters and setters

		public String getInstanceNumber() {
			return instanceNumber;
		}

		public void setInstanceNumber(String instanceNumber) {
			this.instanceNumber = instanceNumber;
		}

		public String getEvent() {
			return event;
		}

		public void setEvent(String event) {
			this.event = event;
		}

		public String getAffectedPosition() {
			return affectedPosition;
		}

		public void setAffectedPosition(String affectedPosition) {
			this.affectedPosition = affectedPosition;
		}

		public String getVotingWeigth() {
			return votingWeigth;
		}

		public void setVotingWeigth(String votingWeigth) {
			this.votingWeigth = votingWeigth;
		}

		public String getWarningSetting() {
			return warningSetting;
		}

		public void setWarningSetting(String warningSetting) {
			this.warningSetting = warningSetting;
		}

		public String getDriftSetting() {
			return driftSetting;
		}

		public void setDriftSetting(String driftSetting) {
			this.driftSetting = driftSetting;
		}

		public String getCreatedOn() {
			return createdOn;
		}

		public void setCreatedOn(String createdOn) {
			this.createdOn = createdOn;
		}

		public String getLastError() {
			return lastError;
		}

		public void setLastError(String lastError) {
			this.lastError = lastError;
		}

		public String getNumberOfClassifiers() {
			return numberOfClassifiers;
		}

		public void setNumberOfClassifiers(String numberOfClassifiers) {
			this.numberOfClassifiers = numberOfClassifiers;
		}

		public String getNumberOfActiveWarnings() {
			return numberOfActiveWarnings;
		}

		public void setNumberOfActiveWarnings(String numberOfActiveWarnings) {
			this.numberOfActiveWarnings = numberOfActiveWarnings;
		}

		public String getClassifiersOnWarning() {
			return classifiersOnWarning;
		}

		public void setClassifiersOnWarning(String classifiersOnWarning) {
			this.classifiersOnWarning = classifiersOnWarning;
		}

		public String getListOfApplicableConcepts() {
			return listOfApplicableConcepts;
		}

		public void setListOfApplicableConcepts(String listOfApplicableConcepts) {
			this.listOfApplicableConcepts = listOfApplicableConcepts;
		}

		public String getRecurringDriftToClassifierID() {
			return recurringDriftToClassifierID;
		}

		public void setRecurringDriftToClassifierID(String recurringDriftToClassifierID) {
			this.recurringDriftToClassifierID = recurringDriftToClassifierID;
		}

		public String getDriftToClassifierCreatedOn() {
			return driftToClassifierCreatedOn;
		}

		public void setDriftToClassifierCreatedOn(String driftToClassifierCreatedOn) {
			this.driftToClassifierCreatedOn = driftToClassifierCreatedOn;
		}

	}
}

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
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;
// import weka.gui.beans.Clusterer;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.igngsvm.gng.GNG;
import moa.classifiers.igngsvm.gng.GUnit;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.DynamicWindowClassificationPerformanceEvaluator;
import moa.evaluation.LearningPerformanceEvaluator;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;

/**
 * Evolving Pool of Classifiers with History
 *
 * @author Andres Leon Suarez Cetrulo (suarezcetrulo at gmail dot com)
 * @version $Revision: 1 $
 * IMP: EPCH deals with structured streams that do not vary their number of features overtime.
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

	public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'x',
			"Change detector for drifts and its parameters", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-5");

	public ClassOption warningDetectionMethodOption = new ClassOption("warningDetectionMethod", 'p',
			"Change detector for warnings (start training bkg learner)", ChangeDetector.class,
			"ADWINChangeDetector -a 1.0E-4");

	public FlagOption disableDriftDetectionOption = new FlagOption("disableDriftDetection", 'u',
			"Should the algorithm use drift detection? If disabled then bkg learner is also disabled.");

	public FlagOption disableRecurringDriftDetectionOption = new FlagOption("disableRecurringDriftDetection", 'r',
			"Should the algorithm save old learners to compare against in the future? If disabled then recurring concepts are not handled explicitly.");

	public IntOption defaultWindowOption = new IntOption("defaultWindow", 'd',
			"Number of rows by default in Dynamic Sliding Windows.", 50, 1, Integer.MAX_VALUE);

	public IntOption minWindowSizeOption = new IntOption("minWindowSize", 'z',
			"Minimum window size in Dynamic Sliding Windows.", 25, 1, Integer.MAX_VALUE);

	public FlagOption resizeAllWindowsOption = new FlagOption("resizeAllWindows", 'b',
			"Should the internal evaluator windows for old learners be dynamic as well?");

	public StringOption eventsLogFileOption = new StringOption("eventsLogFile", 'e',
			"File path to export events as warnings and drifts", "./EPCH_events_log.txt");

	public FlagOption disableEventsLogFileOption = new FlagOption("disableEventsLogFile", 'g',
			"Should export event logs to analyze them in the future? If disabled then events are not logged.");

	public IntOption logLevelOption = new IntOption("eventsLogFileLevel", 'h',
			"0 only logs drifts; 1 logs drifts + warnings; 2 logs every data example", 1, 0, 2);

	public ClassOption evaluatorOption = new ClassOption("baseClassifierEvaluator", 'f',
			"Classification performance evaluation method in each base classifier for voting.",
			LearningPerformanceEvaluator.class, "BasicClassificationPerformanceEvaluator");

	public IntOption warningWindowSizeThresholdOption = new IntOption("WarningWindowSizeThreshold", 'i',
			"Threshold for warning window size that disables a warning.", 1000, 1, Integer.MAX_VALUE);

	public FloatOption distThresholdOption = new FloatOption("distThreshold", 't',
			"Max distance allowed between topologies to be considered part of the same group.", 0.01, 0.000000000000001, Float.MAX_VALUE);
	
	public IntOption minTopologySizeForDriftOption = new IntOption("minTopologySizeForDrift", 'a',
			"Minimum number of prototypes created before allowing a drift.", 1, 1, Integer.MAX_VALUE);
	
	// BY DEFAULT, SAME THAN minWindowSizeOption, AS BOTH SHOULD BE A MIN AMOUNT OF INSTACES THAT REPRESENTS WELL ENOUGH THE SET.
	public IntOption minWSizeForDriftOption = new IntOption("minWSizeForDriftOption", 'w',
			"Minimum number of instances in warning window W before allowing a drift.",
			minWindowSizeOption.getMinValue(), 1, Integer.MAX_VALUE);
	
	public FlagOption updateGroupTopologiesOption = new FlagOption("updateGroupTopologiesOption", 'j',
			"Should the topologies of groups be updated when inserting a new classifier into them? If disabled these won't be updated.");
	
	// Plain-vanilla parameters for the clustering algorithm   // TODO: make it a Clusterer.class
	public ClassOption topologyLearnerOption = new ClassOption("topologyLearner", 'c', "Clusterer to train.", GNG.class,
			"GNG -l 100 -m 200 -a 0.5 -d 0.995 -e 0.2 -n 0.006 -c 100 -b");
	
	// public FloatOption stopPercentageOption = new FloatOption("stopPercentageOption", 'P',
	//		"Stopping criteria as percentage (if 0, the static stopping criteria is )", 0, 0, 100.0);

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
	int conceptsSeen;
	int groupsSeen;
	boolean isOnWarningWindow;  // TODO: this flag could be removed, as the size of W is enough to know this.
	
	protected boolean debug_internalev = false;
	
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
		this.conceptsSeen = 0;
		this.groupsSeen = 0;

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
	 * - 7.1 Check for warning only if the warning window is active.
	 * - 7.1.1 Otherwise update the topology (this is done as long as there is no active warnings).
	 * - Step 7.2 Update warning detection adding latest error
	 * - Step 7.2.0 Check if there was a change (warning signal). If so, start warning window;
	 * - Step 7.2.1 Update the warning detection object for the current object.
	 * 			  This effectively resets changes made to the object while it was still a bkglearner.
	 * - Step 7.2.2 Start warning window.
	 * - 7.3 Check for drift
	 * - 8: Log training event
	 *
	 * The method below implements the following lines of the algorithm:
	 * - Line 1: start topology
	 * - Lines 2-3: initialize ensemble (create base classifiers) and lists.
	 * The rest of the lines of the algorithm are triggered from here
	 * - Line 4-6: ClassifierTrain(c, x, y) -> // Train c on the current instance (x, y).
	 * - Lines 14-15 if warning detection is disabled, update this topology
	 * - Line 16-19: If a warning is detected, start warning window and clear buffer W.	 *
	 * - Lines 7-20: warningDetection:
	 * - Lines 22-33: driftDetection:
	 */
	@Override
	public void trainOnInstanceImpl(Instance instance) {
		++this.instancesSeen;
        if(this.active == null || (!this.disableDriftDetectionOption.isSet() && this.topology == null))
    			init(instance); // Step 0: Initialization

		// Step 1: Update error in concept history learners
		if (!this.disableRecurringDriftDetectionOption.isSet() && this.CH != null
				&& this.CH.getWarnings().containsValue(true) && this.CH.size() > 0) {
			this.CH.updateHistoryErrors(instance);
		} // Steps 2-4: Iterate through the ensemble for following steps (active and bkg classifiers)
		updateEvaluators(instance);
		
		// Step 5: Train base classifier (Lines 4-6)
		this.active.trainOnInstance(instance, this.instancesSeen);
		//System.out.print("before: ");
		// this.CH.printTopologies();  // debug
		// System.out.println("Instances seen: " + this.instancesSeen);  // debug
		// System.out.println("this top size:" + this.topology.getNumberOfPrototypesCreated());

		// Step 6: Check for drifts and warnings only if drift detection is enabled
		if (!this.disableDriftDetectionOption.isSet()) {
			boolean correctlyClassified = this.correctlyClassifies(instance);
			if (this.isOnWarningWindow) trainDuringWarningImpl(instance); // Step 7.1: update actions on warning window
			else this.topology.trainOnInstanceImpl(instance); // Step 7.1.1: Lines 14-15: otherwise train topology
			// System.out.println(this.instancesSeen+" "+this.isOnWarningWindow);
			
			// Update the WARNING detection method
			this.warningDetectionMethod.input(correctlyClassified ? 0 : 1);
			if (this.warningDetectionMethod.getChange()) initWarningWindow(); // Step 7.2.2: new warning?
			
			// Update the DRIFT detection method and check if there was a change: line 22-23 drift detected?
			this.driftDetectionMethod.input(correctlyClassified ? 0 : 1);
			if (this.driftDetectionMethod.getChange()) driftHandlingImpl(correctlyClassified); // Step 7.3: new drift?
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
	public double[] getVotesForInstance(Instance instance) { // although just one active learner. so only this one votes. legacy code from RCARF
		Instance testInstance = instance.copy();
		if (this.active == null) init(testInstance);
		DoubleVector combinedVote = new DoubleVector();
		DoubleVector vote = new DoubleVector(this.active.getVotesForInstance(testInstance));
		if (vote.sumOfValues() > 0.0) {
			vote.normalize();
			//double acc = this.active.evaluator.getPerformanceMeasurements()[1].getValue();
			//if (!this.disableWeightedVote.isSet() && acc > 0.0) {
			//	for (int v = 0; v < vote.numValues(); ++v) vote.setValue(v, vote.getValue(v) * acc);
			//}
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
        List<Measurement> measurementList = new LinkedList<Measurement>();
        measurementList.add(new Measurement("Change detected", this.numberOfDriftsDetected));
        measurementList.add(new Measurement("Warning detected", this.numberOfWarningsDetected));
        // TODO. add all columns from logEvents object, to remove this object
        this.numberOfDriftsDetected = 0;
        this.numberOfWarningsDetected = 0;
        return measurementList.toArray(new Measurement[measurementList.size()]);
    }
	
	private void init(Instance instance) {
		if (!this.disableDriftDetectionOption.isSet() && this.topology == null) {
			this.topology = new Topology(this.topologyLearnerOption); // algorithm line 1
			this.topology.resetLearningImpl();
		}
		if (this.active == null) { // algorithm lines 2-3
			// Init the ensemble.
			BasicClassificationPerformanceEvaluator classificationEvaluator = (BasicClassificationPerformanceEvaluator)
			      getPreparedClassOption(this.evaluatorOption);

			// Only initialize the Concept History if recurring concepts are enabled
			if (!this.disableRecurringDriftDetectionOption.isSet()) CH = new ConceptHistory(this.distThresholdOption.getValue());
			this.W = (Instances) instance.copy().dataset();
			this.W.delete();
			this.isOnWarningWindow = false;

			// START: TO BE REMOVED ONCE THE LOGS ARE NO LONGER REQUIRED
			try { // Start events logging and print headers
				if (this.disableEventsLogFileOption.isSet()) {
					this.eventsLogFile = null;
				} else {
					this.eventsLogFile = new PrintWriter(this.eventsLogFileOption.getValue());
					logEvent(getEventHeaders());
				}
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} // END -TO BE REMOVED

			Classifier learner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
			learner.resetLearning();

			this.active = new EPCHBaseLearner(0, // active classifier pos in an ensemble of active classifiers
					(Classifier) learner.copy(),
					(BasicClassificationPerformanceEvaluator) classificationEvaluator.copy(), this.instancesSeen,
					!this.disableDriftDetectionOption.isSet(), // these are still needed con the level below
					false, // is bkg?
					!this.disableRecurringDriftDetectionOption.isSet(),
					false, // first classifier is not in the CH.
					new Window(this.defaultWindowOption.getValue(),
							1, // size of increments/decrements
							this.minWindowSizeOption.getValue(),
							0.65, // "Decision threshold for recurring concepts (-1 = threshold option disabled)."
							false, // rememberConceptWindowOption?
							this.resizeAllWindowsOption.isSet() ? true : false,
							0), // "Policy to update the size of the window."
					null); // internalEvaluator window starts as NULL
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
	 * - Lines 7-10: Disable warnings after a certain period of time. False warnings handling at buffer W level
	 * - Line 11: if (size(W) 𝝐 (𝟏, 𝝁)) ->In warning window
	 * - Line 12: Train the background classifier
	 * - Line 13: Add instance to the buffer of instances during warning
	 *
	 * The steps followed for this can be seen below:
	 * - Step 1 Disable warning if a length threshold is reached - Lines 7-11
	 * - Step 2 If the classifier is in the warning window, train the bkg classifier and add the current instance to W.
	 *
	 */
	protected void trainDuringWarningImpl(Instance inst) {
		// Step 1: Check if the warning window should be disabled (Lines 7-10)
		// TODO: we could also disable warnings by using a signal to noise ratio variation of the overall classifier during warning
		if (this.W.numInstances() > this.warningWindowSizeThresholdOption.getValue()) resetWarningWindow(true); // Line 8
		// TODO: is this reset necessary? is there any scenario where this causes issues as reseted just before drift?
		else { 		 	
			// Step 2: Either warning window training/buffering or topology update (Lines 11-15)
			this.active.bkgLearner.classifier.trainOnInstance(inst); // Line 11
			this.W.add(inst.copy()); // Line 12
			
			// DEBUG
			if (debug_internalev) {
				System.out.println("------------------");
				System.out.println("W size: "+W.size());
				System.out.println("Internal window size: "+this.active.internalWindowEvaluator.getCurrentSize(0));
				System.out.println("BKG Internal window size: "+this.active.bkgLearner.internalWindowEvaluator.getCurrentSize(0));
				System.out.println("------------------");
			}
		}
		// System.out.println("SIZE OF W: "+ W.numInstances());
	}
	
	protected void resetWarningWindow(boolean resetBkgLearner){
		if (resetBkgLearner) {
			this.active.bkgLearner = null; // Lines 8 and 18
			this.active.internalWindowEvaluator = null;
			this.active.tmpCopyOfClassifier = null;
			System.out.println("BKG AND TMP CLASSIFIERS AND INTERNAL WINDOW RESTARTED");
		}
		this.warningDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.warningDetectionMethodOption)).copy(); // restart warning
		System.out.println("W size was: "+W.numInstances());
		this.W.delete(); // Lines 9 and 19 (it also initializes the object W)
		this.isOnWarningWindow = false;
		if (!this.disableRecurringDriftDetectionOption.isSet()) this.CH.decreaseNumberOfWarnings(0); // update applicable concepts
	}
	
	/**
	 * This starts the warning window event
	 *
	 * The next steps are followed:
	 * - 0 Reset warning
	 * - 1 Update last error and make a backup of the current classifier in a concept object
	 * 		(the active one will be in use until the Drift is confirmed).
	 * - 2 Update of objects with warning.
	 * - 3 If the concept internal evaluator has been initialized for any other classifier on warning,
	 * 		add window size and last error of current classifier on warning.
	 *		Otherwise, initialize a new internal evaluator for the concept
	 * */
	protected void initWarningWindow() {
		this.lastWarningOn = this.instancesSeen;
		this.numberOfWarningsDetected++;
		
		// Step 0 (Lines 17-19)
    resetWarningWindow(true);

		// Step 1 Update last error and make a backup of the current classifier
		if (!this.disableRecurringDriftDetectionOption.isSet()) {
			this.active.saveCurrentConcept(this.instancesSeen); // line 18: Save a tmp copy of c as snapshot
		}
		// Step 2: Update of objects with warning.
		if (!disableRecurringDriftDetectionOption.isSet()) {
			this.active.createInternalEvaluator();
			this.CH.increaseNumberOfWarnings(0, this.active, this.lastError); // 0 is always the pos in an ensemble of only 1 active learner
		}
		if (this.eventsLogFile != null && this.logLevelOption.getValue() >= 1) logEvent(getWarningEvent()); // Log this

		// Step 3: Create background Classifier
		this.active.createBkgClassifier(this.lastWarningOn); // line 19: create background classifier
		this.isOnWarningWindow = true;
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
	 *     This step belong to line 29 in the algorithm: c = FindClassifier(c, b, GH) -> Assign best transition to next state.
	 * - 2 Orchestrate all the actions if the drift is confirmed and there is not a false alarm.
	 * - 4 Decrease amount of warnings in concept history and from evaluators
	 * - 3 reset base learner
	 *
	 * Lines of the algorithm Lines 22-25 are implemented here:
	 * -----------
	 * Insertion in CH (Lines 24-28)
	 * 	line 24: get prototypes from topology
	 * 	line 25: Group for storing old state
	 * 	line 26-27: create a new group represented by 'tmpPrototypes' from Tc
	 * Retrieval from CH and refresh (lines 28-32)
	 * 	line 28: push current classifier to Gc
	 * 	line 29: Update topology on Gc
	 * 	line 30: Group for retrieval of next state (selectDrift()) - In method switchActiveClassifier
	 * 	line 31: Add the examples during warning to a new topology.
	 * 	line 32: reset list W and warning flag
	 */
	protected void driftHandlingImpl(boolean correctlyClassified) {
		this.lastDriftOn = this.instancesSeen;
		this.numberOfDriftsDetected++;
		boolean falseAlarm = false; // Set false alarms (case 1) at false as default

		if (this.topology.getNumberOfPrototypesCreated() >= minTopologySizeForDriftOption.getValue()
				&& this.W.numInstances() >= minWSizeForDriftOption.getValue()) {
			System.out.println(); System.out.println(); System.out.println();
			System.out.println(); System.out.println(); System.out.println();
			System.out.println("DRIFT DETECTED!");
			System.out.println("The topology has been trained (before the warning) with "+this.topology.getLearner().getInstancesSeen()+" instances so far.");

			// Retrieval from CH
			if (!this.disableRecurringDriftDetectionOption.isSet()) { // step 1: lines 29-31 of the algorithm
				// Start retrieval from CH (TODO: W should only be used if warning detection is enabled. same for topologies?)
				falseAlarm = switchActiveClassifier(this.CH.findGroup(this.W)); // TODO: refactor so false alarm is checked by an specific method
			} else if (this.eventsLogFile != null && this.logLevelOption.getValue() >= 1)  // TODO: remove this.eventsLogFile != null as a condition
				logEvent(getBkgDriftEvent());  // TODO. Refactor this logEvent function so it's inside of 'registerDrift' and not wrapping it
			if (!falseAlarm) {  // Step 2
				// Insertion in CH (Lines 23-26)
				Instances tmpPrototypes = this.topology.getPrototypes(); // line 23
				System.out.println();
				System.out.println("Pc size:" + this.topology.getNumberOfPrototypesCreated());
				int previousTopologyGroupId = this.CH.findGroup(tmpPrototypes); // line 24
				if (this.CH.size() == 0 || previousTopologyGroupId == -1) { // line 25
					previousTopologyGroupId = this.groupsSeen++;
					System.out.println("CREATING NEW GROUP: " + previousTopologyGroupId); // TODO: remove once debugged. check that numbers are added correctly
					this.CH.createNewGroup(previousTopologyGroupId, this.topology.clone()); //, this.newTopology); // line 27
				} else System.out.println("SELECTING FROM GROUP: " + previousTopologyGroupId); // TODO: remove once debugged. check that numbers are selected correctly
							
				// Move copy of active classifier made before warning to Concept History.
				this.active.tmpCopyOfClassifier.addHistoryID(this.conceptsSeen++);
				this.CH.addLearnerToGroup(previousTopologyGroupId, this.active.tmpCopyOfClassifier); // line 28  // debug. check that this is not a duplicate. that it's a separate object.

				if (updateGroupTopologiesOption.isSet()) { // not in v5 of algorithm (line 29 in v4)
					// OJO: esta linea de abajo necesita agregar .clone/.copy en algun sitio, ya que no esta creando objetos nuevos y  hace que el updatetopology de abajo actualice el objeto de grupo de CH
					// this.CH.setGroupTopology(previousTopologyGroupId,
					//		updateTopology(this.CH.copyTopologyFromGroup(previousTopologyGroupId), this.topology.getPrototypes()));
					// asuarez 02-06.2019. quiza esto incluso fuese mas adecuado
					this.CH.setGroupTopology(previousTopologyGroupId, this.topology.clone()); // pero entonces perderiamos mucho historico..
				}

				this.active.reset(); // reset base classifier and transition to bkg or recurring learner (step 3)
				// Reset warning related params (but not the learners as these are used at the next step)
				if (this.newTopology != null) this.topology = updateTopology(this.newTopology.clone(), this.W); // line 31
				/* System.out.println("**ññññññññ***");
				System.out.println(this.topology.getNumberOfPrototypesCreated());
				if (this.newTopology != null)
					System.out.println(this.newTopology.getNumberOfPrototypesCreated());
				else
					System.out.println("null");
				System.out.println("**ññññññññ***");*/
				resetWarningWindow(false); // step 4 and line 32. false as argument as the bkg related params are reseted in the prior line.
			}
		} else {
			System.out.println("There weren't enough prototypes in the topology or instances in W.");
			registerDriftFalseAlarm();
		}
		System.out.println("Pn size:" + this.topology.getNumberOfPrototypesCreated());
		// Reset drift independently on both false alarm and actual drift cases.
		this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy();
		this.newTopology = null;
		/*System.out.print("just after creation: ");
		this.CH.printTopologies();
		System.out.println();*/
	}

	// DRIFT ACTIONS

	/***
	 * Register false alarm an update variables consequently
	 * (both the active classifier and the ensemble topology will remain being the same)
	 */
	protected boolean registerDriftFalseAlarm() {
		if (this.eventsLogFile != null && this.logLevel >= 0) logEvent(getFalseAlarmEvent()); // TODO: refactor
		// this.newTopology = null; // then Pn = Pc  (line 30)  // not needed as null by default in drifthandling method
		return true;
	}

	/***
	 * Register recurring drift an update variables consequently
	 * Copy the best recurring learner in the history group passed, and the topology of this group.
	 */
	protected void registerRecurringDrift(Integer indexOfBestRanked, int historyGroup) {
		if (this.eventsLogFile != null && this.logLevel >= 0)
			logEvent(getRecurringDriftEvent(indexOfBestRanked, historyGroup));  // TODO: refactor
		this.active.bkgLearner = this.CH.copyConcept(historyGroup, indexOfBestRanked);
		this.newTopology = this.CH.copyTopologyFromGroup(historyGroup);  // then Pn = Ph  (line 30)
		this.newTopology.resetId();
	}

	/***
	 * Register background drift an update variables consequently
	 * Pselected is a new P in case of background drift
	 */
	protected void registerBkgDrift() {
		// Register background drift
		if (this.eventsLogFile != null && this.logLevel >= 0) logEvent(getBkgDriftEvent());  // TODO: refactor
		this.newTopology = this.topology.clone(); // line 30  (new Topology does not create a new object due to the meta class for the clusterer)
		this.newTopology.resetId();
		this.newTopology.resetLearningImpl(); // line 30
	}

	// IDENTIFICATION OF NEXT STATE

	/**
	 * This method ranks all applicable base classifiers in the Concept History (CH)
	 * It also selects the next classifier to be active, or it raises a false alarm
	 * if the drift should be reconsidered.
	 *
	 * It implements the line 30 of the algorithm.
	 *  get topology of the group (methods 'registerDrift...') and retrieve the best classifier
	 *
	 * -----------------------------------------------------------------------------------------------------------------
	 * False alarms depend on the drift decision mechanism
	 * -----------------------------------------------------------------------------------------------------------------
	 *
	 * False alarms are taken into consideration for drifts (the warning will be still active
	 * even if a false alarm is raised for a drift in the same active classifier).
	 * If the background learner is NULL, we consider that the drift signal may have been caused
	 * by a too sensitive drift detection parameterization. In this case, it's clearly
	 * too soon to change the active classifier. Therefore we raise a drift signal.
	 *
	 * We also raise a false alarm when the active classifier obtains less error than the
	 * bkg classifier and all of the classifiers from the CH.
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
	protected boolean switchActiveClassifier(int historyGroup) {  // TODO: refactor so false alarm is checked by an specific method
	    int indexOfBestRanked = -1;
		double errorOfBestRanked = -1.0;
		HashMap<Integer, Double> ranking = new HashMap<Integer, Double> ();
		
		// 1 Raise a false alarm for the drift if the background learner is not ready
		if (this.active.bkgLearner == null) return registerDriftFalseAlarm();
		    
		// 2 Retrieve best applicable classifier from Concept History (if a CH group applies)
		if (historyGroup != -1) ranking = rankConceptHistoryClassifiers(historyGroup);
		if (ranking.size() > 0) {
			indexOfBestRanked = getMinKey(ranking); // find index of concept with lowest value (error)
			errorOfBestRanked = Collections.min(ranking.values());
		}
		// 3 Compare this against the background classifier and make the decision.
		if (activeBetterThanBKGbaseClassifier()) {
			if (ranking.size() > 0 && !activeBetterThanCHbaseClassifier(errorOfBestRanked))
				registerRecurringDrift(indexOfBestRanked, historyGroup);
			// False alarm if active classifier is still the best one and when there are no applicable concepts.
			else return registerDriftFalseAlarm();
		} else {
			if (ranking.size() > 0 && bkgBetterThanCHbaseClassifier(errorOfBestRanked))
				registerRecurringDrift(indexOfBestRanked, historyGroup);
			else registerBkgDrift();
			
		} return false; // No false alarms raised at this point
	}

	/**
	 * This auxiliary function updates either old or new topologies that will be merged with, compared with, or will replace to the current one.
	 * */
	protected Topology updateTopology(Topology top, Instances w2) {
				
		/* If there is a different stopping criteria, it should depend on the avg error obtained by the topologies
		 * as at some point in the learning process GNG will only improve the accuracy a bit, and slowly, as the approximation should get better overtime.
		 * Stopping early should only give us speed (loosing accuracy). If we do this (we may not need to), the tradeoff is the key.
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
			// We add them once
		for (int instPos = 0; instPos < w2.numInstances(); instPos++) {
			top.trainOnInstanceImpl(w2.get(instPos).copy());
			// System.out.println("prototypes created in topology:" + top.getNumberOfPrototypesCreated());
		}
		// }
		return top; // if topology (Pc) is global, then we don´t need to return this here
	}
		
	/* Compute distances between Instances as seen in GNG for arrays (GUnit objects). **/
	public double computeDistance(Instance w1,Instance w2){
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
		// DEBUG
		System.out.println("ACTIVE: "+((DynamicWindowClassificationPerformanceEvaluator) this.active.internalWindowEvaluator)
			.getFractionIncorrectlyClassified(this.active.indexOriginal));
		System.out.println("BKG: "+((DynamicWindowClassificationPerformanceEvaluator)
				this.active.bkgLearner.internalWindowEvaluator).getFractionIncorrectlyClassified(this.active.bkgLearner.indexOriginal));
		System.out.println("activeBetterThanBKGbaseClassifier: "+(((DynamicWindowClassificationPerformanceEvaluator) this.active.internalWindowEvaluator)
			.getFractionIncorrectlyClassified(this.active.indexOriginal) <= ((DynamicWindowClassificationPerformanceEvaluator)
			this.active.bkgLearner.internalWindowEvaluator).getFractionIncorrectlyClassified(this.active.bkgLearner.indexOriginal)));
		
		return (((DynamicWindowClassificationPerformanceEvaluator) this.active.internalWindowEvaluator)
			.getFractionIncorrectlyClassified(this.active.indexOriginal) <= ((DynamicWindowClassificationPerformanceEvaluator)
			this.active.bkgLearner.internalWindowEvaluator).getFractionIncorrectlyClassified(this.active.bkgLearner.indexOriginal));
		// this.indexOriginal - pos of this classifier with active warning in ensemble
		// return ((this.ensemble.evaluator.getFractionIncorrectlyClassified() <= ((DynamicWindowClassificationPerformanceEvaluator)
		//		this.ensemble.bkgLearner.internalWindowEvaluator).getFractionIncorrectlyClassified(this.ensemble.bkgLearner.indexOriginal)));
	}

	protected boolean activeBetterThanCHbaseClassifier(double bestFromCH) {
		//DEBUG
		System.out.println("ACTIVE: "+((DynamicWindowClassificationPerformanceEvaluator) this.active.internalWindowEvaluator)
				.getFractionIncorrectlyClassified(this.active.indexOriginal));
			System.out.println("CH: "+bestFromCH);
			System.out.println("activeBetterThanCHbaseClassifier: "+(((DynamicWindowClassificationPerformanceEvaluator) this.active.internalWindowEvaluator)
					.getFractionIncorrectlyClassified(this.active.indexOriginal) <= bestFromCH));
			
		return (((DynamicWindowClassificationPerformanceEvaluator) this.active.internalWindowEvaluator)
			.getFractionIncorrectlyClassified(this.active.indexOriginal) <= bestFromCH);

		// this.indexOriginal - pos of this classifier with active warning in ensemble
		// return (this.ensemble.evaluator.getFractionIncorrectlyClassified() <= bestFromCH); (old comparison)
	}

	protected boolean bkgBetterThanCHbaseClassifier(double bestFromCH) {
		// DEBUG
		System.out.println("ACTIVE: "+((DynamicWindowClassificationPerformanceEvaluator) this.active.bkgLearner.internalWindowEvaluator)
				.getFractionIncorrectlyClassified(this.active.bkgLearner.indexOriginal));
		System.out.println("CH: "+bestFromCH);
		System.out.println("bkgBetterThanCHbaseClassifier: "+(bestFromCH <= ((DynamicWindowClassificationPerformanceEvaluator) this.active.bkgLearner.internalWindowEvaluator)
				.getFractionIncorrectlyClassified(this.active.bkgLearner.indexOriginal)));
	
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
		System.out.println(indexOfBestRankedInCH); // TODO: remove after debugging
		System.out.println("RECURRING DRIFT RESET IN POSITION #"+0+" TO MODEL #"+
				CH.get(group).groupList.get(indexOfBestRankedInCH).historyIndex + " FROM GROUP "+group);
		String[] eventLog = { String.valueOf(this.lastDriftOn), "RECURRING DRIFT", String.valueOf(0),
				String.valueOf(this.active.evaluator.getPerformanceMeasurements()[1].getValue()),
				this.warningDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				this.driftDetectionMethodOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				String.valueOf(this.instancesSeen), String.valueOf(this.active.evaluator.getFractionIncorrectlyClassified()),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings().size(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getNumberOfActiveWarnings(): "N/A"),
				String.valueOf(!this.disableRecurringDriftDetectionOption.isSet() ? this.CH.getWarnings(): "N/A"),
				"N/A",
				String.valueOf(this.CH.get(group).groupList.get(indexOfBestRankedInCH).historyIndex),
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

		private void init(int indexOriginal, Classifier classifier,
				BasicClassificationPerformanceEvaluator evaluatorInstantiated, long instancesSeen, // boolean useBkgLearner,
				boolean useDriftDetector, boolean isBackgroundLearner, boolean useRecurringLearner, boolean isOldLearner,
				Window windowProperties, DynamicWindowClassificationPerformanceEvaluator internalEvaluator) {

			this.indexOriginal = indexOriginal;
			this.createdOn = instancesSeen;

			this.classifier = classifier;
			this.evaluator = evaluatorInstantiated;
			
			this.useDriftDetector = useDriftDetector;
			
			this.isBackgroundLearner = isBackgroundLearner;

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
				long instancesSeen, boolean useDriftDetector, boolean isBackgroundLearner,boolean useRecurringLearner, boolean isOldLearner,
				Window windowProperties, DynamicWindowClassificationPerformanceEvaluator bkgInternalEvaluator) {
			init(indexOriginal, classifier, evaluatorInstantiated, instancesSeen, useDriftDetector, isBackgroundLearner,
					useRecurringLearner, isOldLearner, windowProperties, bkgInternalEvaluator);
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
			if (this.bkgLearner != null) {
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
			this.internalWindowEvaluator.setEvaluatorType("ACTIVE");  // for debugging
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
			BasicClassificationPerformanceEvaluator bkgEvaluator = (BasicClassificationPerformanceEvaluator) this.evaluator.copy();
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
				bkgInternalWindowEvaluator.setEvaluatorType("BKG");  // for debugging
				bkgInternalWindowEvaluator.reset();
			} System.out.println("------------------------------");

			// 4 Create a new bkgLearner object
			this.bkgLearner = new EPCHBaseLearner(indexOriginal, bkgClassifier, bkgEvaluator, lastWarningOn, this.useDriftDetector,
					true, this.useRecurringLearner, false, this.windowProperties, bkgInternalWindowEvaluator);
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
		*		and the first warning arises. See it in initWarningWindow
		*  Step 3 Add the classifier accumulated error (from the start of the classifier) from the iteration before the warning
		*  A simple concept to be stored in the concept history that doesn't have a running learner.
		*  This doesn't train. It keeps the classifier as it was at the beginning of the training window to be stored in case of drift.
		*/
		public void saveCurrentConcept(long instancesSeen) {
			// step 1
			this.lastError = this.evaluator.getFractionIncorrectlyClassified();
			// step 2
			EPCHBaseLearner tmpConcept = new EPCHBaseLearner(this.indexOriginal, this.classifier.copy(),
					(BasicClassificationPerformanceEvaluator) this.evaluator.copy(), this.createdOn, this.useDriftDetector,
					true, useRecurringLearner, true, this.windowProperties.copy(), null);
			// step 3
			this.tmpCopyOfClassifier = new Concept(tmpConcept, this.createdOn,
					this.evaluator.getPerformanceMeasurements()[0].getValue(), instancesSeen);
			this.tmpCopyOfClassifier.setErrorBeforeWarning(this.lastError);
		}
	}

	protected class Concept {
		protected int ensembleIndex; // position that it had in the ensemble when this supports many active classifiers.
		protected int historyIndex; // id in concept history

		// Stats
		protected long createdOn;
		protected long instancesSeen;
		protected double classifiedInstances;
		protected double errorBeforeWarning;

		// Learner
		public EPCHBaseLearner ConceptLearner;  // TODO: should this be a meta learner?

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
			tmpInternalWindow.setEvaluatorType("CH");
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
	protected class Topology implements Cloneable {
		
		protected GNG learner; // TODO: this should be a meta class of cluster instead
		private Instance auxInst;
		private int id;
		
		public Topology(ClassOption topologyLearnerOption) {
			this.learner = (GNG) getPreparedClassOption(topologyLearnerOption);  // TODO: make it a generic clustering algorithm (not casted as GNG)
			this.id = (int) (Math.random() * 1000);
		}
		
		protected void resetId() {
			this.id = (int) (Math.random() * 1000);
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
			return ((GNG) this.learner).getNumberOfPrototypesCreated();
		}
		
		// method with bug
		protected Instances prototypesToUnsupervisedInstances(ArrayList<GUnit> tmpPrototypes) {
			Instances tmp = new Instances ((this.auxInst).dataset());
			tmp.delete();
			Instance inst = null;

			for(GUnit prototype: tmpPrototypes) {
				inst = (Instance) this.auxInst.copy();
				for (int j = 0; j < prototype.w.length; j++) inst.setValue(j, prototype.w[j]);
				tmp.add(inst);
			}
			return tmp;
		}
		
		public Instances getPrototypes() {
			return prototypesToUnsupervisedInstances(this.learner.getS());
		}
		
		public void setLearner (GNG newLearner) {
			this.learner = (GNG) newLearner;
		}
		
		public GNG getLearner () {
			return this.learner;
		}
		
		@Override
		protected Topology clone() {
		    try {
		    		Topology cloned = (Topology) super.clone();
		    		cloned.setLearner(cloned.getLearner().clone());
				return cloned;
			} catch (CloneNotSupportedException e) {
				e.printStackTrace();
			} return null;
		}
	}
	
	protected class Group {
		int id;
		Topology conceptCluster;
		public ConcurrentHashMap<Integer, Concept> groupList; // List of concepts per group

		public Group(int id, Topology top) {
			this.id = id; // nextID();
			this.conceptCluster = top;
			this.groupList = new ConcurrentHashMap<Integer, Concept>();
		}

		// method changed on 01-06-2019
		public Group(int id, ClassOption topologyLearnerOption) {
			this.id = id; // nextID();
			this.conceptCluster = new Topology(topologyLearnerOption);
			this.groupList = new ConcurrentHashMap<Integer, Concept>();
		}
		
		// method added on 01-06-2019
		public void init(Instances tmpPrototypes) {
			for (int instPos = 0; instPos < tmpPrototypes.size(); instPos++) {
				this.conceptCluster.trainOnInstanceImpl(tmpPrototypes.get(instPos));
			}
		}

		public EPCHBaseLearner copyConcept(int key) {
			EPCHBaseLearner aux = this.groupList.get(key).getBaseLearner();
			return aux;
		}

		// Getters
		public int getID() {
			return this.id;
		}

		public Instances getTopologyPrototypes() {
			return this.conceptCluster.getPrototypes(); // these instances don't belong to a class (unsupervised)
		}
		
		public void setConceptCluster(Topology topology2) {
			this.conceptCluster = topology2;
		}
		
		public Topology getConceptCluster() {
			return this.conceptCluster;
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
		protected boolean debugNN = true;

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
		
		public void printTopologies() {
			for (Group g : this.history.values()) {
				System.out.println("*******************");
				System.out.println("Learner ID: "+g.getConceptCluster().getLearner().id);
				System.out.println("TOP ID: "+g.getConceptCluster().id);
				System.out.println(g.getTopologyPrototypes().numInstances()+" prototypes in group "+g.getID());
				System.out.println("*******************");
			}
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
		public void decreaseNumberOfWarnings(int ensemblePos) {
			this.classifiersOnWarning.put(ensemblePos, false);
			if (this.history != null && this.history.size() > 0) {
				// This adds it as an applicable concept to all the groups, as we don't know its group yet
				// TODO: performance improvement: can we have only internal evaluators between classifiers inside a given group and not across groups?
				for (int historyGroup : history.keySet()) {
					for (Concept learner : this.history.get(historyGroup).values()) {
						// TODO: performance improvement: can we use use oldGroup instead of historyGroup? (then change it also when increasing warnings)
						if (learner.getInternalEvaluator() != null && learner.getInternalEvaluator().containsIndex(ensemblePos)) {
							learner.deleteModelFromInternalEvaluator(ensemblePos);
							if (learner.getAmountOfApplicableModels() == 0) learner.setInternalEvaluator(null);  // TODO: implement this in RCARF & Evolving RCARF once tested
						}
					}
				}
			}
		}
		
		/** Creation of new group and pushing of this to the CH
		protected void createNewGroup(int groupId, Instances tmpPrototypes, ClassOption learnerOption) { // , Topology newTopology) {
			Group g = new Group(groupId, learnerOption);
			g.init(tmpPrototypes);
			this.history.put(groupId, g); // the id is there twice to keep track of it and for testing purposes.
		}*/
		protected void createNewGroup(int groupId,  Topology top) {
			Group g = new Group(groupId, top);
			
			System.out.println();
			System.out.println("SAVING TOPOLOGY:" + g.conceptCluster.getNumberOfPrototypesCreated());
			
			this.history.put(groupId, g); // the id is there twice to keep track of it and for testing purposes.
		}
		
		/**
		 * This method receives the current list of training examples received during
		 * the warning window and checks what's the closest group.
		 */
		protected int findGroup(Instances w1) {
			double min = Double.MAX_VALUE;
			double dist = 0;
			int group = -1;

			// asuarez: Debugging groups
			System.out.println();
			System.out.println("There were "+w1.numInstances()+" instances for comparison.");
			System.out.println("There are "+this.history.size()+" groups.");
			for (Group g : this.history.values()) {
				dist = getMeanDistanceToNN(w1, g.getTopologyPrototypes()) / w1.numInstances();
				if (dist < min) {
					min = dist;
					group = g.getID();
				}
				System.out.println("Distance of instances in W to group "+g.getID()+" is "+dist);
				System.out.println("Learner ID: "+g.getConceptCluster().getLearner().id);
				System.out.println(g.getTopologyPrototypes().numInstances()+" prototypes in group "+g.getID());
				System.out.println("Number of historical classifiers in the group: "+g.groupList.size());
				System.out.println("--");
			}
			System.out.println("");

			if (dist < this.maxDistanceThreshold) {
				System.out.println("The selected group, therefore, is group:"+group);
				return group;
			} else {
				System.out.println("No groups selected as the distance is "+dist+" and the maximum dist allowed is: "+this.maxDistanceThreshold);
				return -1;
			}
		}
		
		/**
		 * This function computes the sum of the distances between every prototype in
		 * the topology passed and the current list of instances during warning (W)
		 */
		/*protected double getMeanDistance(Instances w1, Instances w2) {
			double [] dist = new double[w1.numInstances()];
			double totalDist = 0.0;
			for (int instPos1 = 0; instPos1 < w1.numInstances(); instPos1++) {
				for (int instPos2 = 0; instPos2 < w2.numInstances(); instPos2++) {
					dist[instPos1] += computeDistance(w1.get(instPos1), w2.get(instPos2));
				} dist[instPos1] = dist[instPos1] / w2.numInstances();
			} 
			// Averaging distances
			for (int i = 0; i < dist.length; i++)  totalDist += dist[i];
			return totalDist / w1.numInstances();
		}*/
		
		/**
		 * This function computes the sum of the distances between the nearest prototype in a group's topology 
 		 *  and the current list of instances during warning (W)
		 */
		protected double getMeanDistanceToNN(Instances w1, Instances topologyPrototypes) {
			int nPrototypes = 1; // number of neighbors
			Instances nearestPrototypes;
			System.out.println("\n%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%\n");
			try {
				double [] dist = new double[w1.numInstances()];
				double totalDist = 0.0;
				for (int i = 0; i < w1.numInstances(); i++) {
					nearestPrototypes = getNearestPrototypes(w1.get(i), topologyPrototypes, nPrototypes);
					for (int j = 0; j < nearestPrototypes.numInstances(); j++) {
						dist[i] += computeDistance(w1.get(i), nearestPrototypes.get(j)); // squared distance (default in GNG)
					} dist[i] = dist[i] / nearestPrototypes.numInstances(); // divided by 1 if only 1 neighbour (default)
					
					if (this.debugNN) assert nearestPrototypes.numInstances() == nPrototypes; 
				} 
				// Averaging distances
				for (int i = 0; i < dist.length; i++)  totalDist += dist[i];
				return totalDist / w1.numInstances();
				
			} catch (Exception e) {
				e.printStackTrace();
			}  
			return (Double) null; 
		}

		public Instances getNearestPrototypes(Instance i, Instances topologyPrototypes, int nPrototypes) throws Exception {
			// Set converters to use WEKAS filters
			SamoaToWekaInstanceConverter converter = new SamoaToWekaInstanceConverter();
			WekaToSamoaInstanceConverter deconverter = new WekaToSamoaInstanceConverter();
			
			// Instantiate NN search object and return results 
			NearestNeighbourSearch m_NNSearch = new LinearNNSearch();
		    m_NNSearch.setInstances(converter.wekaInstances(topologyPrototypes));
		    //TODO: distances using a kernel => m_NNSearch.getDistances();
		    return deconverter.samoaInstances(m_NNSearch.kNearestNeighbours(converter.wekaInstance(i), nPrototypes));
		    
		    // Instances neighbours = deconverter.samoaInstances(m_NNSearch.kNearestNeighbours(converter.wekaInstance(i), nPrototypes));
			// if (this.debugNN) System.out.println("SIZE AFTER NN: "+neighbours.numInstances());
			// if (this.debugNN) System.out.println("TSIZE BEFORE NN:  "+topologyPrototypes.numInstances());
		    // return neighbours;
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
		
		public Topology copyTopologyFromGroup(int key) {
			return this.get(key).getConceptCluster().clone();
		}
				
		public void setGroupTopology(int groupID, Topology top) {
			this.history.get(groupID).setConceptCluster(top); // line 26
		}
		
		public void addLearnerToGroup(int groupID, Concept learner) {
			this.history.get(groupID).put(learner.historyIndex, learner); // line 28
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

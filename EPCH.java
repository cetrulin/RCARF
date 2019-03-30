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

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.DoubleVector;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.options.ClassOption;
import weka.gui.sql.event.HistoryChangedListener;

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
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;

import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.DynamicWindowClassificationPerformanceEvaluator;
import moa.evaluation.LearningPerformanceEvaluator;

import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.meta.EPCH.ConceptHistory;
import moa.classifiers.meta.EPCH.Event;

//for EPCH
import moa.classifiers.igngsvm.gng.GNG;  
import moa.classifiers.igngsvm.gng.GUnit;


/**
 * Evolving Pool of Classifiers with History
 *
 * @author Andres Leon Suarez Cetrulo (suarezcetrulo at gmail dot com)
 * @version $Revision: 1 $
 */
public class EPCH extends AbstractClassifier implements MultiClassClassifier {

    @Override
    public String getPurposeString() {
        return "EPCH from Suarez-Cetrulo et al.";
    }
    
    private static final long serialVersionUID = 1L;

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree -e 1000000 -g 200 -c 0"); // default params for hoeffding trees
    
    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
        "The number of trees.", 1, 1, Integer.MAX_VALUE);
    
    public FloatOption lambdaOption = new FloatOption("lambda", 'a',
        "The lambda parameter for bagging.", 6.0, 1.0, Float.MAX_VALUE);

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
	// Added in previous versions by @suarezcetrulo
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
    		"Should the comparison windows for old learners be also dynamic?");
    
    public StringOption eventsLogFileOption = new StringOption("eventsLogFile",'e',"File path to export events as warnings and drifts", "./EPCH_events_log.txt");
    
    public FlagOption disableEventsLogFileOption = new FlagOption("disableEventsLogFile", 'g', 
            "Should export event logs to analyze them in the future? If disabled then events are not logged.");
   
    public IntOption logLevelOption = new IntOption("eventsLogFileLevel", 'h', 
            "0 only logs drifts; 1 logs drifts + warnings; 2 logs every data example", 1, 0, 2);

    public ClassOption evaluatorOption = new ClassOption("baseClassifierEvaluator", 'f',
            "Classification performance evaluation method in each base classifier for voting.",
            LearningPerformanceEvaluator.class,
            "BasicClassificationPerformanceEvaluator");
    
    public IntOption driftDecisionMechanismOption = new IntOption("driftDecisionMechanism", 'k', 
            "0 does not take into account the performance active base classifier explicitely, at the time of the drift; 1 takes into consideration active classifiers", 
            0, 0, 2);
    
    public IntOption warningWindowSizeThresholdOption = new IntOption("WarningWindowSizeThreshold", 'h', 
            "Threshold for warning window size that defines a false a alarm.", 300, 1, Integer.MAX_VALUE);
	
    protected EPCHBaseLearner[] ensemble;
    protected long instancesSeen;
    protected int subspaceSize;
    protected BasicClassificationPerformanceEvaluator evaluator;
    
    // Window statistics
    protected double lastError;

    // Warning and Drifts
    public long lastDriftOn;	
    public long lastWarningOn;
    
    // The drift and warning object parameters. 
    protected ClassOption driftOption;
    protected ClassOption warningOption;	
    
    // Drift and warning detection
    protected ChangeDetector driftDetectionMethod;
    protected ChangeDetector warningDetectionMethod;	
    
    protected int numberOfDriftsDetected;
	protected int numberOfWarningsDetected;	
	
	PrintWriter eventsLogFile;
    public int logLevel;

	GNG topology;
	GNG newTopology;
    ArrayList<Instance> W = new ArrayList<Instance>();  // list of training examples during warning window. 

    
    @Override
    public void resetLearningImpl() {
        // Reset attributes
        this.ensemble = null;
        this.subspaceSize = 0;
        this.instancesSeen = 0;
        this.topology = null;
                
        // Reset warning and drift detection related attributes
        this.lastDriftOn = 0;	
        this.lastWarningOn = 0; 
        
        this.numberOfDriftsDetected = 0;	
        this.numberOfWarningsDetected = 0;	
        
        // (in the place where the base learners are created), so there is only one of these.
        if(!this.disableDriftDetectionOption.isSet()) {
            //this.driftOption = driftOption;  no longer required
            this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftOption)).copy();
        }

        // Init Drift Detector for Warning detection. 
        if(!this.disableBackgroundLearnerOption.isSet()) {
            //this.warningOption = warningOption;  no longer required
            this.warningDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.warningOption)).copy();
        }
                
        this.evaluator = new BasicClassificationPerformanceEvaluator();
        
    }

    @Override
	/** 
	 * In EPCH, this method performs the actions of the classifier manager. 
	 * Thus, in this method, warning and drift detection are performed.
	 * This method also send instances to the ensemble classifiers. 
	 *  Or to the single active classifier if ensemble size = 1 (default).
	 *  
	 *  New BKG classifiers and switching from and to CH may also need to be here.
	 * */
    public void trainOnInstanceImpl(Instance instance) {    	
        ++this.instancesSeen;
        
        if(this.topology == null)
    			topology = new GNG();  // line 2
        		topology.resetLearningImpl();
		    	/*
		    	topology.nParadaOption=this.nParadaOption; // TODO. add input option for this?
		    	topology.lambdaOption = this.lambdaOption;
		    	topology.alfaOption = this.alfaOption;
		    	topology.maxAgeOption = this.maxAgeOption;
		    	topology.constantOption = this.constantOption;
		    	topology.BepsilonOption = this.BepsilonOption;
		    	topology.NepsilonOption = this.NepsilonOption; 
		    	*/
        		        
        if(this.ensemble == null) 
            initEnsemble(instance); // line 1
        
        // 1 If the concept history is ready and it contains old classifiers, testing in each old classifier internal evaluator (to compare against bkg one)
        if (!disableRecurringDriftDetectionOption.isSet() && ConceptHistory.historyList != null && ConceptHistory.classifiersOnWarning.containsValue(true) && ConceptHistory.historyList.size() > 0) {
	        for (int historyGroup: ConceptHistory.historyList.keySet()) {
	        		for (Concept oldClassifier : ConceptHistory.historyList.get(historyGroup).groupList.values()) { 
		            DoubleVector oldClassifierVote = new DoubleVector(oldClassifier.ConceptLearner.getVotesForInstance(instance)); 
	        			if (oldClassifier.ConceptLearner.internalWindowEvaluator != null && 
	        					oldClassifier.ConceptLearner.internalWindowEvaluator.getAmountOfApplicableModels() > 0) { 
	        				// When the concept is added the first time, it doesn't have applicable classifiers. They are not inserted until the first warning. 
	        				// So the Concept History only runs over warning windows
	        				oldClassifier.ConceptLearner.internalWindowEvaluator.addResult(new InstanceExample(instance), oldClassifierVote.getArrayRef());
	        			}
		        	}
        		}
        } 
        
        for (int i = 0 ; i < this.ensemble.length ; i++) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(instance));
            InstanceExample example = new InstanceExample(instance);
            
            // 2 Testing in active classifier
            this.ensemble[i].evaluator.addResult(example, vote.getArrayRef());

            if(!disableRecurringDriftDetectionOption.isSet()) {
	            // 3 If the warning window is open, testing in background classifier internal evaluator (for comparison purposes) 
	            if(this.ensemble[i].bkgLearner != null && this.ensemble[i].bkgLearner.internalWindowEvaluator!=null 
	            		&& this.ensemble[i].bkgLearner.internalWindowEvaluator.containsIndex(this.ensemble[i].bkgLearner.indexOriginal)) {
	                 DoubleVector bkgVote = new DoubleVector(this.ensemble[i].bkgLearner.getVotesForInstance(instance)); 
	            		this.ensemble[i].bkgLearner.internalWindowEvaluator.addResult(example, bkgVote.getArrayRef());
	            }
            }
		   trainOnInstance(i, instance, this.instancesSeen);
        }    
    }

    public void trainOnInstance(int ensemblePos, Instance instance, long instancesSeen) {
	    	this.ensemble[ensemblePos].trainOnInstance(instance, instancesSeen); // Line 6: ClassifierTrain(c, x, y) -> Train c on the current instance (x, y).     
	    
	    // Should it use a drift detector? Also, is it a backgroundLearner? If so, then do not "incept" another one. 
	    if(!this.disableDriftDetectionOption.isSet()) { // && !this.ensemble[ensemblePos].isBackgroundLearner)  THIS SHOULDN'T BE NECESSARY
	    		System.out.println("This is a background learner but it shouldn't! "+this.ensemble[ensemblePos].isBackgroundLearner); // just check
	    		// same check below. it should crash if it doesnt pass. if this happens, then the uncommented condition above may be necessary
	    		assert !this.ensemble[ensemblePos].isBackgroundLearner; 
	        boolean correctlyClassifies = this.ensemble[ensemblePos].correctlyClassifies(instance);
	        		
	        // Check for warning only if useBkgLearner is active
	        if(!this.disableBackgroundLearnerOption.isSet()) { 
	            /*********** warning detection ***********/
	            // Update the WARNING detection method
	            this.warningDetectionMethod.input(correctlyClassifies ? 0 : 1);
	            
	            // TODO (confirm the next lines 7-15)
	            if (this.W.size() > warningWindowSizeThresholdOption.getValue()) this.W.clear();	 // Lines 7-9:  False Alarm (Case 2)					
	            if (this.W.size() > 1 && this.W.size() < warningWindowSizeThresholdOption.getValue()) { // Lines 10-15 if (size(W) 𝝐 (𝟏, 𝝁)) ->In warning window
	            		if(this.ensemble[ensemblePos].bkgLearner != null) 
	            			this.ensemble[ensemblePos].bkgLearner.classifier.trainOnInstance(instance);	// Line 11: Train the background classifier
	            		this.W.add(instance);
	            }
	            else UpdateTopology(instance);  // Line 15:  Update centroids / prototypes.
	            
	            // Check if there was a change – in case of false alarm this triggers warning again and the bkglearner gets replaced
	            if(this.warningDetectionMethod.getChange()) {  // line 16: warning detected?
	                raiseWarning(ensemblePos, instancesSeen);
	                this.W.add(instance);	// line 20
	            } 
	            
	        } /*********** drift detection ***********/
	        // Update the DRIFT detection method
	        this.driftDetectionMethod.input(correctlyClassifies ? 0 : 1);
	        // Check if there was a change
			if(this.driftDetectionMethod.getChange()) {  // line 22 drift detected?                		
				raiseDrift(ensemblePos, instancesSeen, this.eventsLogFile, this.logLevelOption.getValue());
			}
			        
	    } if (this.eventsLogFile != null && logLevelOption.getValue() >= 2) logEvent(getTrainExampleEvent(ensemblePos)); // Register training example in log

    }
    	    
	// Starts Warning window
	public void startWarningWindow(int ensemblePos) {
		// 0 Reset warning window
		this.ensemble[ensemblePos].bkgLearner = null;   // not 'this', the base classifier. but now warnings are at ensemble level. how do we handle this?
	    if(! disableRecurringDriftDetectionOption.isSet()) {    	        	
	        	this.ensemble[ensemblePos].internalWindowEvaluator = null;
		        
		    // 1 Updating objects with warning. Turns on windows flag in Concept History.
		    // Also, if the concept history is ready and it contains old classifiers, add prior estimation and window size to each concepts history learner
			ConceptHistory.classifiersOnWarning.put(this.ensemble[ensemblePos].indexOriginal, true);
		    if(ConceptHistory.historyList != null && ConceptHistory.historyList.size() > 0) {
			    for (int historyGroup: ConceptHistory.historyList.keySet()) {
			        	for (Concept oldClassifier : ConceptHistory.historyList.get(historyGroup).groupList.values()) { // TODO: figure this out
			        		// If the concept internal evaluator has been initialized for any other classifier on warning, add window size and last error of current classifier on warning 
			        		if (oldClassifier.ConceptLearner.internalWindowEvaluator != null) {
			        			//// System.out.println("ADDING VALUES TO INTERNAL EVALUATOR OF CONCEPT "+oldClassifier.historyIndex+" IN POS "+this.indexOriginal);
				        		((DynamicWindowClassificationPerformanceEvaluator) 
				        			oldClassifier.ConceptLearner.internalWindowEvaluator).addModel(ensemblePos,
				        					this.lastError,
				        					this.ensemble[ensemblePos].windowProperties.windowSize);
			        		} 
			        		// Otherwise, initialize a new internal evaluator for the concept
			        		else {
			        			//// System.out.println("INSTANCIATING FOR THE FIRST TIME INTERNAL EVALUATOR FOR CONCEPT "+oldClassifier.historyIndex+" IN POS "+this.indexOriginal);
			       			// TODO: is this correct? can we simplify this in any way?
			        			DynamicWindowClassificationPerformanceEvaluator tmpInternalWindow = new DynamicWindowClassificationPerformanceEvaluator(
			       				this.ensemble[ensemblePos].windowProperties.getSize(), 
			       				this.ensemble[ensemblePos].windowProperties.getIncrements(), 
			       				this.ensemble[ensemblePos].windowProperties.getMinSize(),
			            			this.ensemble[ensemblePos].lastError, 
			            			this.ensemble[ensemblePos].windowProperties.getDecisionThreshold(),
			            			this.ensemble[ensemblePos].windowProperties.getDynamicWindowInOldClassifiersFlag(), 
			            			this.ensemble[ensemblePos].windowProperties.getResizingPolicy(),
			            			this.ensemble[ensemblePos].indexOriginal, 
			            			"created for old-retrieved classifier in ensembleIndex #"+this.ensemble[ensemblePos].indexOriginal);  	
			       			tmpInternalWindow.reset(); 
			       			
			       			oldClassifier.ConceptLearner.internalWindowEvaluator = tmpInternalWindow;
			        		}
			        	}
		        	}
		    } 
	    } // Log warning     
	    if (eventsLogFile != null && logLevelOption.getValue() >= 1) logEvent(getWarningEvent(ensemblePos));
	    
		// 2 Create background Classifier
		this.ensemble[ensemblePos].createBkgClassifier(this.lastWarningOn);	// line 18: create background classifier
		W.clear();	 // line 19: (TODO: correct?) 
		
		// Update the warning detection object for the current object 
		// (this effectively resets changes made to the object while it was still a bkg learner). 
		this.warningDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.warningOption)).copy();
	}
	

	private void raiseWarning(int ensemblePos, long instancesSeen) {
		this.lastWarningOn = instancesSeen;
		// TODO: update here 'lastWarningOn' flag in classifiers of the ensemble, or in another entity
		this.numberOfWarningsDetected++;

		// 1 Update last error and make a backup of the current classifier in a concept object (the active one will be in use until the Drift is confirmed). 
		// As there is no false alarms explicit mechanism (bkgLeaners keep running till replaced), this has been moved here.
		if(!this.disableRecurringDriftDetectionOption.isSet()) this.ensemble[ensemblePos].saveCurrentConcept(instancesSeen);  // line 17: Save a tmp copy of c as snapshot
		// TODO. call (saveCurrentConcept) of the baseClassifier
		   
		// 2 Start warning window to create bkg learner and retrieve old classifiers (if option enabled)
		startWarningWindow(ensemblePos);  // lines 18-19
	} 
		
	/**
	 * Aux method to select the next concept classifier and topology when a drift is raised
	 * Pselected is:
		a new P (Pn) in case of bkgDrift;
		Pc in case of false alarm; 
		and Ph in case of recurring drift.
	 * */
	private void raiseDrift(int ensemblePos, long instancesSeen, PrintWriter eventsLogFile, int logLevel) {
		this.lastDriftOn = instancesSeen;
		this.numberOfDriftsDetected++;
		//toLog = false;
		
	    // Set false alarms in case of drift at false as default
	    boolean falseAlarm = false; // Included for cases where driftDecisionMechanism > 0 and recurring drifts are enabled.
					
  		// 1 Compare DT results using Window method and pick the best one between concept history and bkg classifier.
  		// It returns the best classifier in the object of the bkgLearner
		//  if there is not another base classifier with lower error than active classifier (and driftDecisionMechanism > 0), then a false alarm is raised
	    	if (!this.disableRecurringDriftDetectionOption.isSet()) falseAlarm = switchActiveClassifier(ensemblePos);  // line 32:  c = FindClassifier(c, b, GH) -> Assign best transition to next state
		else if (eventsLogFile != null && logLevel >= 1 ) logEvent(getBkgDriftEvent(ensemblePos)); 
		
		// 2 Transition to new classifier (only if there is no false alarms)
		if (!falseAlarm) this.ensemble[ensemblePos].reset(); 
		
		// asuarez. 25 sept 2018
		// IMPORTANT. False alarms avoid drifts, but they do not disable a certain warning on a base classifier. // any TODO here??
		// In cases of a false alarm, the active classifiers will still have an active warning.
		
	}

	 /** 
         *  This method ranks all applicable base classifiers in the Concept History (CH)
         *  It also selects the next classifier to be active, or it raises a false alarm if the drift should be reconsidered.
         * 
		 *  -----------------------------------------------------------------------------------------------------------------
         *  False alarms depend on the drift decision mechanism
		 *  -----------------------------------------------------------------------------------------------------------------
		 *  
		 * 	When driftDecisionMechanism == 0, if bkgLearner == null, false alarms cannot be raised. A comparison against CH is not possible as there is no bkg learner trained.
	     *	 In this case, a drift signal has been raised and it cannot be stopped without false alarms. A bkg drift applies as only option available.
		 *  
         * 	When drift decision mechanism == 1 or 2, then false alarms are taken into consideration for drifts (the warning will be still active even if a false alarm is raised for a drift in the same active classifier).
	    	 *	If the background learner is NULL, we consider that the drift signal may have been caused by a too sensitive drift detection parameterization.
	    	 *	In this case, it's clearly too soon to change the active classifier. Therefore we raise a drift signal. 
	     *		
	     *	When drift decision mechanism == 2, we also raise a false alarm when the active classifier obtains less error than the bkg classifier and all of the classifiers from the CH.
	     *		
	     *  -----------------------------------------------------------------------------------------------------------------
		 *	If the active classifier is not the best available choice / false alarm is raised, the following logic applies:
		 *  -----------------------------------------------------------------------------------------------------------------
		 *  If bkgBetterThanCHbaseClassifier == False, the minimum error of the base classifiers in the CH is not lower than the error of the bkg classifier. 
		 *   Then, register background drift.
		 *   
		 *	If CHranking.size() == 0, no applicable concepts for the active classifier in the concept history. Then, we register background drift.
	    	 *	Otherwise, a recurring drift is the best option. 
         * @param historyGroup 
	    	 *			    		 
         * */
        public boolean switchActiveClassifier(int ensemblePos) { // TODO: verify that ensemblePos makes sense here. maybe not??
			// TODO: confirm use the input parameter 'historyGroup', that refers to the most similar group of classifiers in the Concept History
			
        		// Start retrieval from CH 
        		int historyGroup = findGroup(instancesToArray(W)); // line 31: Group for retrieval of next state (TODO)
		
			// 1 Raise a false alarm for the drift if the background learner is not ready
			if (this.driftDecisionMechanismOption.getValue() > 0 && this.ensemble[ensemblePos].bkgLearner == null) return registerDriftFalseAlarm(ensemblePos);
			
			// 2 Retrieve best applicable classifier from Concept History
			int indexOfBestRanked = -1;
			double errorOfBestRanked = -1.0;
			HashMap<Integer, Double> ranking = rankConceptHistoryClassifiers(ensemblePos, historyGroup);
			if (ranking.size() > 0) {
				indexOfBestRanked = getMinKey(ranking); // find index of concept with lowest value (error)
				errorOfBestRanked = Collections.min(ranking.values());
			}
			
			// 3 Compare this against the background classifier and make the decision.
			if (this.driftDecisionMechanismOption.getValue() == 2) {
				if (activeBetterThanBKGbaseClassifier(ensemblePos)) { 
					if (ranking.size() >0 && !activeBetterThanCHbaseClassifier(errorOfBestRanked))
								 registerRecurringDrift(ensemblePos, indexOfBestRanked, historyGroup); 
						// false alarm if active classifier is still the best one and when there are no applicable concepts.
						else return registerDriftFalseAlarm(ensemblePos); 
				} else { 
						if(ranking.size() > 0 && bkgBetterThanCHbaseClassifier(ensemblePos, errorOfBestRanked)) 
							registerRecurringDrift(ensemblePos, indexOfBestRanked, historyGroup);
						else registerBkgDrift(ensemblePos); 					
				}
				
			// Drift decision mechanism == 0 or 1 (in an edge case where the bkgclassifier is still NULL, we ignore the comparisons)
			} else {
				if(ranking.size() > 0 && this.ensemble[ensemblePos].bkgLearner != null && bkgBetterThanCHbaseClassifier(ensemblePos, errorOfBestRanked)) 
					registerRecurringDrift (ensemblePos, indexOfBestRanked, historyGroup);
				else registerBkgDrift (ensemblePos);
			} 
			
			return false; // No false alarms raised at this point
	}

    public ArrayList<double[]> instancesToArray(ArrayList<Instance> WInstances){ // TODO. Maybe we should use the class Instances for this
    		ArrayList<double[]> converted = new ArrayList<double[]>();
		for(Instance inst: WInstances) { 
			converted.add(inst.toDoubleArray());	// TODO: Make sure that we are adding the class value as a feature in GNG
		} return converted;
    }
	/**
	 * This method receives the current list of training examples received 
	 * during the warning window and checks what's the closest group.
	 * */
	private int findGroup(ArrayList<double[]> arrayList) {
		double min = Double.MAX_VALUE;
		double dist = 0;
		int group = -1;
		
		for (Group g: ConceptHistory.historyList.values()) {
			dist = getAbsoluteSumDistances(arrayList, g.getTopologyPrototypes());
			if (dist < min) {                                               
				min = dist;
				group = g.getID();
			}		
		}
		if (dist < this.warningWindowSizeThresholdOption.getValue()) { 
			return group;
		} else return -1;		
	}
	
	/**
	 * This function computes the sum of the distances between every prototype 
	 * in the topology passed and the current list of instances during warning (W)
	 * *
	private double getAbsoluteSumDistances(ArrayList<Instance> WInstances, ArrayList<GUnit> topologyPrototypes, String type) {
		double totalDist = 0.0;	
		
		for(Instance inst: WInstances) { 
			for (GUnit prototype: topologyPrototypes) {
				totalDist += GUnit.dist(prototype.w, inst.toDoubleArray());	// TODO: Make sure that we are adding the class value as a feature in GNG
			}
		} return totalDist;
	}*/
	
	/**
	 * This function computes the sum of the distances between every prototype 
	 * in the topology passed and the current list of instances during warning (W)
	 * */
	private double getAbsoluteSumDistances(ArrayList<double[]> WInstances, ArrayList<double[]> topologyPrototypes) { // TODO: make sure that the metric used here makes sense
		double totalDist = 0.0;	
		
		for(double[] inst: WInstances) { 
			for (double[] prototype: topologyPrototypes) {
				totalDist += GUnit.dist(prototype, inst);	// TODO: Make sure that we are adding the class value as a feature in GNG
			}
		} return totalDist;
	}

	// this.indexOriginal - pos of this classifier with active warning in ensemble
	public HashMap<Integer, Double> rankConceptHistoryClassifiers (int ensemblePos, int historyGroup) {
			HashMap<Integer, Double> CHranking = new HashMap<Integer, Double>();
			// Concept History owns only one learner per historic concept. But each learner saves all classifier's independent window size and priorEstimation in a HashMap.
			for (Concept auxConcept : ConceptHistory.historyList.get(historyGroup).groupList.values()) 
				// Only take into consideration Concepts sent to the Concept History after the current classifier raised a warning (see this consideration in reset*) 
				if (auxConcept.ConceptLearner.internalWindowEvaluator != null && 
						auxConcept.ConceptLearner.internalWindowEvaluator.containsIndex(ensemblePos)) { // checking indexOriginal to verify that it's an applicable concept
					CHranking.put(auxConcept.getHistoryIndex(), ((DynamicWindowClassificationPerformanceEvaluator) 
							auxConcept.ConceptLearner.internalWindowEvaluator).getFractionIncorrectlyClassified(ensemblePos)); // indexOriginal refers to the classifier we compare against. it should be an applicable concept.
				}
			
			return CHranking;
	}
	
	// Aux method for getting the best classifier in a hashMap of (int classifierIndex, double averageErrorInWindow) 
	private Integer getMinKey(Map<Integer, Double> map) {
		Integer minKey = null;
		double minValue = Double.MAX_VALUE;
		for(Integer key : map.keySet()) {
			double value = map.get(key);
			if(value < minValue) {
				minValue = value;
				minKey = key;
			}
		} return minKey;
	}
	
	// this.indexOriginal - pos of this classifier with active warning in ensemble
	public boolean activeBetterThanBKGbaseClassifier(int ensemblePos) { 
			// TODO? We may need to do use an internal evaluator for the active learner when driftDecisionMechanism==2. 
			//		 But the resizing mechanism may need to be different, or compare to the bkg learner.
			/* return (((DynamicWindowClassificationPerformanceEvaluator) this.internalWindowEvaluator)
						.getFractionIncorrectlyClassified(this.indexOriginal) <= 
					((DynamicWindowClassificationPerformanceEvaluator) this.bkgLearner.internalWindowEvaluator)
						.getFractionIncorrectlyClassified(this.bkgLearner.indexOriginal));*/
		
		return (( this.ensemble[ensemblePos].evaluator.getFractionIncorrectlyClassified() <= 
					((DynamicWindowClassificationPerformanceEvaluator) this.ensemble[ensemblePos].bkgLearner.internalWindowEvaluator)
					.getFractionIncorrectlyClassified(this.ensemble[ensemblePos].bkgLearner.indexOriginal))); 
		// this.bkgLearner.indexOriginal == this.indexOriginal, as it is created for that same ensemble pos.  
	}
	
	// this.indexOriginal - pos of this classifier with active warning in ensemble
	public boolean activeBetterThanCHbaseClassifier(double bestFromCH) {			
			// TODO? We may need to do use an internal evaluator for the active learner when driftDecisionMechanism==2. 
			//		 But the resizing mechanism may need to be different, or compare to the bkg learner.
			/* return (((DynamicWindowClassificationPerformanceEvaluator) this.internalWindowEvaluator)
					.getFractionIncorrectlyClassified(this.indexOriginal) <= bestFromCH);*/
		
		return (this.evaluator.getFractionIncorrectlyClassified() <= bestFromCH);
	}

	// this.bkgLearner.indexOriginal - pos of bkg classifier if it becomes active in the ensemble
	public boolean bkgBetterThanCHbaseClassifier(int ensemblePos, double bestFromCH) {
		return (bestFromCH <= ((DynamicWindowClassificationPerformanceEvaluator) this.ensemble[ensemblePos].bkgLearner.internalWindowEvaluator)
				.getFractionIncorrectlyClassified(this.ensemble[ensemblePos].bkgLearner.indexOriginal));
		// this.bkgLearner.indexOriginal == this.indexOriginal, as it is created for that same ensemble pos.  
	}

	public boolean registerDriftFalseAlarm (int ensemblePos) {
		// Register false alarm.
		if (this.eventsLogFile != null && this.logLevel >= 0 ) logEvent(getFalseAlarmEvent(ensemblePos));	
		this.newTopology = null;  // then Pn = Pc  // TODO: confirm
		// this.newPrototypes = getPrototypes(newTopology);  // line 33   (TODO: confirm that this is not necessary)
		
		// Update false alarm (the active classifier will remain being the same)
		return true;
	}
	
	public void registerRecurringDrift (int ensemblePos, Integer indexOfBestRanked, int historyGroup) {
		// Register recurring drift
		if (this.eventsLogFile != null && this.logLevel >= 0 ) logEvent(getRecurringDriftEvent(indexOfBestRanked, historyGroup, ensemblePos));	
		
		// Extracts best recurring learner from concept history. It no longer exists in the concept history
		this.ensemble[ensemblePos].bkgLearner = ConceptHistory.copyConcept(historyGroup, indexOfBestRanked);  // TODO: replace with the row below
		// this.bkgLearner = ConceptHistory.extractConcept(historyGroup, indexOfBestRanked); // TODO: this should come from a Group
		// this.newTopology = ConceptHistory.getGroupTopology(historyGroup); // TODO:
		// this.newPrototypes = getPrototypes(newTopology);  // line 33   (TODO: confirm that this is not necessary)
	}
	
	public void registerBkgDrift (int ensemblePos) {
		// Register background drift
		if (this.eventsLogFile != null && this.logLevel >= 0 ) logEvent(getBkgDriftEvent(ensemblePos));
		// this.newTopology = new GNG();		// TODO
		// newTopology.train(W);				// TODO
		// this.newPrototypes = getPrototypes(newTopology);  // line 33   (TODO: confirm that this is not necessary)
	}
	
    // Auxiliar methods for logging events
    
    public Event getTrainExampleEvent(int indexOriginal) {
    		String [] eventLog = {
	        String.valueOf(instancesSeen), "Train example", String.valueOf(indexOriginal), // TODO: not THIS. in the base classifier
			String.valueOf(this.evaluator.getPerformanceMeasurements()[1].getValue()), 
			this.warningOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), 
			this.driftOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), 
			String.valueOf(this.instancesSeen), String.valueOf(this.evaluator.getFractionIncorrectlyClassified()), 
			String.valueOf(! this.disableRecurringDriftDetectionOption.isSet() ? ConceptHistory.classifiersOnWarning.size() : "N/A"),
			String.valueOf(! this.disableRecurringDriftDetectionOption.isSet() ? ConceptHistory.getNumberOfActiveWarnings() : "N/A"),
			String.valueOf(! this.disableRecurringDriftDetectionOption.isSet() ? ConceptHistory.classifiersOnWarning : "N/A"), "N/A", "N/A", "N/A"};
    		
    		return (new Event(eventLog));
    }
       
    public Event getWarningEvent(int indexOriginal) {
    	
        // System.out.println();
        // System.out.println("-------------------------------------------------");
        // System.out.println("WARNING ON IN MODEL #"+this.indexOriginal+". Warning flag status (activeClassifierPos, Flag): "+ConceptHistory.classifiersOnWarning);
        // System.out.println("CONCEPT HISTORY STATE AND APPLICABLE FROM THIS WARNING IS: "+ConceptHistory.historyList.keySet().toString());
        // System.out.println("-------------------------------------------------");
        // System.out.println();
    	
    		String [] warningLog = {
				String.valueOf(this.lastWarningOn), "WARNING-START", // event
				String.valueOf(indexOriginal), String.valueOf(this.evaluator.getPerformanceMeasurements()[1].getValue()),  
				this.warningOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
				this.driftOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), 
				String.valueOf(this.instancesSeen), String.valueOf(this.evaluator.getFractionIncorrectlyClassified()), 
				String.valueOf(! this.disableRecurringDriftDetectionOption.isSet() ? ConceptHistory.classifiersOnWarning.size() : "N/A"),
				String.valueOf(! this.disableRecurringDriftDetectionOption.isSet() ? ConceptHistory.getNumberOfActiveWarnings() : "N/A"),
				String.valueOf(! this.disableRecurringDriftDetectionOption.isSet() ? ConceptHistory.classifiersOnWarning : "N/A"), 
				! this.disableRecurringDriftDetectionOption.isSet() ? ConceptHistory.historyList.keySet().toString() : "N/A", "N/A", "N/A"};
    		//1279,1,WARNING-START,0.74,{F,T,F;F;F;F},...

    		return (new Event(warningLog));
    }
    
    public Event getBkgDriftEvent(int indexOriginal) {
    	
        // System.out.println("DRIFT RESET IN MODEL #"+this.indexOriginal+" TO NEW BKG MODEL #"+this.bkgLearner.indexOriginal); 

    		String [] eventLog = {String.valueOf(this.lastDriftOn), "DRIFT TO BKG MODEL", String.valueOf(indexOriginal), 
					String.valueOf(this.evaluator.getPerformanceMeasurements()[1].getValue()), 
					this.warningOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), 
					this.driftOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
					String.valueOf(this.instancesSeen), String.valueOf(this.evaluator.getFractionIncorrectlyClassified()), 
					String.valueOf(! this.disableRecurringDriftDetectionOption.isSet() ? ConceptHistory.classifiersOnWarning.size() : "N/A"), 
					String.valueOf(! this.disableRecurringDriftDetectionOption.isSet() ? ConceptHistory.getNumberOfActiveWarnings() : "N/A"), 
					String.valueOf(! this.disableRecurringDriftDetectionOption.isSet() ? ConceptHistory.classifiersOnWarning : "N/A"), 
					"N/A", "N/A", "N/A"};    
			    		
    		return (new Event(eventLog));
    }
    
    public Event getRecurringDriftEvent(Integer indexOfBestRankedInCH, int group, int indexOriginal) {
    		
    		//// System.out.println(indexOfBestRankedInCH); // TODO: debugging
    	        	
        // System.out.println("RECURRING DRIFT RESET IN POSITION #"+this.indexOriginal+" TO MODEL #"+ConceptHistory.historyList.get(indexOfBestRankedInCH).ensembleIndex); //+this.bkgLearner.indexOriginal);   

    		String [] eventLog = {
		        String.valueOf(this.lastDriftOn), "RECURRING DRIFT", String.valueOf(indexOriginal), 
	    		String.valueOf(this.evaluator.getPerformanceMeasurements()[1].getValue()), 
	    		this.warningOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), 
	    		this.driftOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), 
	    		String.valueOf(this.instancesSeen), String.valueOf(this.evaluator.getFractionIncorrectlyClassified()), 
	    		String.valueOf(! this.disableRecurringDriftDetectionOption.isSet() ? ConceptHistory.classifiersOnWarning.size() : "N/A"),
	    		String.valueOf(! this.disableRecurringDriftDetectionOption.isSet() ? ConceptHistory.getNumberOfActiveWarnings() : "N/A"), 
	    		String.valueOf(! this.disableRecurringDriftDetectionOption.isSet() ? ConceptHistory.classifiersOnWarning : "N/A"), "N/A",
	    		String.valueOf(ConceptHistory.historyList.get(group).groupList.get(indexOfBestRankedInCH).ensembleIndex),
	    		String.valueOf(ConceptHistory.historyList.get(group).groupList.get(indexOfBestRankedInCH).createdOn)
    		};    
    		
    		return (new Event(eventLog));
    }
            
    public Event getFalseAlarmEvent(int indexOriginal) {
    	
        // System.out.println("DRIFT RESET IN MODEL #"+this.indexOriginal+" TO NEW BKG MODEL #"+this.bkgLearner.indexOriginal); 

    		String [] eventLog = {String.valueOf(this.lastDriftOn), "FALSE ALARM ON DRIFT SIGNAL", String.valueOf(indexOriginal), 
					String.valueOf(this.evaluator.getPerformanceMeasurements()[1].getValue()), 
					this.warningOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""), 
					this.driftOption.getValueAsCLIString().replace("ADWINChangeDetector -a ", ""),
					String.valueOf(this.instancesSeen), String.valueOf(this.evaluator.getFractionIncorrectlyClassified()), 
					String.valueOf(! this.disableRecurringDriftDetectionOption.isSet() ? ConceptHistory.classifiersOnWarning.size() : "N/A"), 
					String.valueOf(! this.disableRecurringDriftDetectionOption.isSet() ? ConceptHistory.getNumberOfActiveWarnings() : "N/A"), 
					String.valueOf(! this.disableRecurringDriftDetectionOption.isSet() ? ConceptHistory.classifiersOnWarning : "N/A"), 
					"N/A", "N/A", "N/A"};
			    		
    		return (new Event(eventLog));
    }


    @Override
    public double[] getVotesForInstance(Instance instance) {
        Instance testInstance = instance.copy();
        if(this.ensemble == null) 
            initEnsemble(testInstance);
        DoubleVector combinedVote = new DoubleVector();
        
        // TODO: Change this testing/predict function. Do we need voting? Bear in mind that at some point we'll have many classifiers.
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
    
    /** Incremental method 
     * @return */
	private GNG UpdateTopology(Instance inst) { // TODO. are the topologies global variables or we send them as attributes?
			this.topology.trainOnInstanceImpl(inst);  // TODO: Make sure that we send class as a feature, so everything makes sense after.
		return this.topology;
	}

    
    protected void initEnsemble(Instance instance) {
    	    	
        // Init the ensemble.
        int ensembleSize = this.ensembleSizeOption.getValue();
        this.ensemble = new EPCHBaseLearner[ensembleSize];
        BasicClassificationPerformanceEvaluator classificationEvaluator = (BasicClassificationPerformanceEvaluator) getPreparedClassOption(this.evaluatorOption);
  
        // Only initialize Concept History if explicit recurring concepts handling is enabled
        if(!this.disableRecurringDriftDetectionOption.isSet()) {
			ConceptHistory.lastID = 0;
			ConceptHistory.historyList = new ConcurrentHashMap<Integer,Group> ();
			ConceptHistory.classifiersOnWarning = new ConcurrentHashMap<Integer,Boolean> ();
        }
        
        try { // Start events logging and print headers
	    		if (disableEventsLogFileOption.isSet()) {
	    			eventsLogFile = null;
	    		} else {
	    			eventsLogFile = new PrintWriter(this.eventsLogFileOption.getValue());
	    			logEvent(getEventHeaders()); 
	    		}
        	} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
            
        Classifier learner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        learner.resetLearning();
        
        for(int i = 0 ; i < ensembleSize ; ++i) {	
            this.ensemble[i] = new EPCHBaseLearner(
                i, 
                (Classifier) learner.copy(), 
                (BasicClassificationPerformanceEvaluator) classificationEvaluator.copy(), 
                this.instancesSeen, 
                ! this.disableBackgroundLearnerOption.isSet(),  // these are still needed con the level below
                ! this.disableDriftDetectionOption.isSet(), 	// these are still needed con the level below
                // driftDecisionMechanismOption.getValue(),
                //driftDetectionMethodOption,
                //warningDetectionMethodOption,
                false, // isbkglearner
                ! this.disableRecurringDriftDetectionOption.isSet(),
                false, // @suarezcetrulo : first classifier is not old. An old classifier (retrieved from the concept history).
                new Window(this.defaultWindowOption.getValue(), this.windowIncrementsOption.getValue(), this.minWindowSizeOption.getValue(), this.thresholdOption.getValue(), 
        				this.rememberConceptWindowOption.isSet()? true: false, this.resizeAllWindowsOption.isSet()? true: false, windowResizePolicyOption.getValue()),
                null, // @suarezcetrulo : Windows start at NULL
                // eventsLogFile,
                // logLevelOption.getValue(),
                warningWindowSizeThresholdOption.getValue() //,
                // new GNG()
            		);
        }
    }
    
    /**
     * Inner class that represents a single tree member of the ensemble. 
     * It contains some analysis information, such as the numberOfDriftsDetected, 
     */
    protected final class EPCHBaseLearner {
        public int indexOriginal;
        public long createdOn;
        public Classifier classifier;
        public boolean isBackgroundLearner;
        public boolean isOldLearner; // only for reference
        
        public boolean useBkgLearner; // these flags are still necessary at this level
        public boolean useDriftDetector;
        public boolean useRecurringLearner; 
        
        // Bkg learner
        protected EPCHBaseLearner bkgLearner;
        
        // Copy of main classifier at the beginning of the warning window for its copy in the Concept History
        protected Concept tmpCopyOfClassifier;  
        
        // Statistics
        public BasicClassificationPerformanceEvaluator evaluator;
       
        // Internal statistics
        public DynamicWindowClassificationPerformanceEvaluator internalWindowEvaluator; // only used in background and old classifiers
        protected double lastError;  // TODO: now this variable is duplicated. also present on the level above (CM), so watch out!
        protected Window windowProperties;
        
        // TODO: delete the next parameters that are commented out if that is ok.
        // public int driftDecisionMechanism; 
        // public PrintWriter eventsLogFile;
        // public int logLevel;
        
        // new on EPHC
        // GNG newTopology = null;
        // GNG topology = null;
        // ArrayList<GUnit> newPrototypes = null;
        int warningWindowSizeThreshold = -1;
        
        
        private void init(int indexOriginal, Classifier classifier, BasicClassificationPerformanceEvaluator evaluatorInstantiated, 
            long instancesSeen, 
        		boolean useBkgLearner, boolean useDriftDetector, //int driftDecisionMechanism, 
            boolean isBackgroundLearner, boolean useRecurringLearner, boolean isOldLearner, 
            Window windowProperties, DynamicWindowClassificationPerformanceEvaluator internalEvaluator,
            // PrintWriter eventsLogFile, int logLevel, 
            int warningWindowSizeThreshold) { //, GNG topology) { // last parameters added by @suarezcetrulo
        	
            this.indexOriginal = indexOriginal;
            this.createdOn = instancesSeen;
            // this.eventsLogFile = eventsLogFile;
            // this.logLevel = logLevel;
            
            this.classifier = classifier;
            this.evaluator = evaluatorInstantiated;
            
            this.useBkgLearner = useBkgLearner; 
            this.useRecurringLearner = useRecurringLearner; 
            this.useDriftDetector = useDriftDetector; 
            // this.driftDecisionMechanism = driftDecisionMechanism;
            
            this.isBackgroundLearner = isBackgroundLearner;
            
            if (useRecurringLearner) { // !this.disableRecurringDriftDetectionOption.isSet() ? 
                // Window params
                this.windowProperties=windowProperties;
                // Recurring drifts
                this.isOldLearner = isOldLearner;
                // only used in bkg and retrieved old classifiers
                this.internalWindowEvaluator = internalEvaluator;
            }
            
            // for EPCH
            this.warningWindowSizeThreshold = warningWindowSizeThreshold;
            	// this.topology = topology;
        }
        
        public boolean correctlyClassifies(Instance instance) {
			return this.classifier.correctlyClassifies(instance);
		}

		// last inputs parameters added by @suarezcetrulo
        public EPCHBaseLearner(int indexOriginal, Classifier classifier, BasicClassificationPerformanceEvaluator evaluatorInstantiated, 
                    long instancesSeen, 
        			   boolean useBkgLearner, boolean useDriftDetector, // int driftDecisionMechanism,
                    boolean isBackgroundLearner,  boolean useRecurringLearner, boolean isOldLearner, 
                    Window windowProperties, DynamicWindowClassificationPerformanceEvaluator bkgInternalEvaluator, 
                    // PrintWriter eventsLogFile, int logLevel, 
                    int warningWindowSizeThreshold) {  // , GNG topology) {
            init(indexOriginal, classifier, evaluatorInstantiated, instancesSeen, useBkgLearner, useDriftDetector, // driftDecisionMechanism, 
            		isBackgroundLearner, useRecurringLearner, isOldLearner, 
            		 windowProperties, bkgInternalEvaluator, // eventsLogFile, logLevel, 
            		 warningWindowSizeThreshold); // , topology);
        }

        private int createNewGroup(ArrayList<double[]> tmpPrototypes) {
    			int id = ConceptHistory.nextGroupID();
        		Group g = new Group(id, newTopology);
        		ConceptHistory.historyList.put(id, g);  // the id is there twice to keep track of it and for testing purposes.
			return id;
		}
        
        public void reset() {

            // System.out.println();
            // System.out.println("-------------------------------------------------");
            // System.out.println("RESET (WARNING OFF) IN MODEL #"+this.indexOriginal+". Warning flag status (activeClassifierPos, Flag): "+ConceptHistory.classifiersOnWarning);
            // System.out.println("-------------------------------------------------");
            // System.out.println();
		    // Transition to the best bkg or retrieved old learner
        		if (this.useBkgLearner && this.bkgLearner != null) {
        			// userRecurringLearner = ! this.disableRecurringDriftDetectionOption.isSet()
        		   if(this.useRecurringLearner) { // && ConceptHistory.historyList != null && ConceptHistory.historyList.size() > 0) {
        			  // 1 Decrease amount of warnings in concept history and from evaluators
        			  ConceptHistory.classifiersOnWarning.put(this.indexOriginal, false);
        			  if(ConceptHistory.historyList != null && ConceptHistory.historyList.size() > 0) {
        			      for (int historyGroup: ConceptHistory.historyList.keySet()) {
        			    	  	// TODO: does this bit make sense? review. this use to be over only one list, as there were not groups before.
	        				for (Concept oldClassifier: ConceptHistory.historyList.get(historyGroup).groupList.values()) { // figure this out
	        					  if (oldClassifier.ConceptLearner.internalWindowEvaluator != null && 
	        							  oldClassifier.ConceptLearner.internalWindowEvaluator.containsIndex(this.indexOriginal))
	        						  ((DynamicWindowClassificationPerformanceEvaluator) 
	        								  oldClassifier.ConceptLearner.internalWindowEvaluator).deleteModel(this.indexOriginal);
	        				}
        			      }
        	          }
        	            
        			  // TODO. IMPORTANT. does topology exists at this level? see the reference to it below. this may crash. should we move this to the ensemble level or what?
        	          // Insertion in CH (TODO: get all aux methods coded)
        			  ArrayList<double[]> tmpPrototypes = topology.getPrototypes(); // line 24
        	          int group = findGroup(tmpPrototypes); // line 25: Group for storing old state 
        	          if (ConceptHistory.historyList.size() == 0 || group == -1) { // line 26
        	          	group = createNewGroup(tmpPrototypes); // line 27: create a placeholder for a group represented by 'tmpPrototypes' 
        	          }

        			  // 2.1 Move copy of active classifier made before warning to Concept History. Its history ID will be the last one in the history (= size)
        	          this.tmpCopyOfClassifier.addHistoryID(ConceptHistory.nextID());
        	          // ConceptHistory.historyList[Gc_id].put(
        	          //										this.tmpCopyOfClassifier.historyIndex, 
        	          //										this.tmpCopyOfClassifier); // line 29:  push current classifier to Gc (TODO)
        			  this.tmpCopyOfClassifier = null; 	// reset tc.
        			  // Consideration *: This classifier is added to the concept history, but it wont be considered by other classifiers on warning until their next warning.
       			  // If it becomes necessary in terms of implementation for this concept, to be considered immediately by the other examples in warning,
       			  // we could have a HashMap in ConceptHistory with a flag saying if a given ensembleIndexPos needs to check the ConceptHistory again and add window sizes and priorError.

       	          // ConceptHistory.historyList[Gc_id].topology = mergeTopologies (CH[Gc_id], tp); // line 30: Update topology on Gc (TODO)
        			  
        			  // TODO: Important! This probably may need to be at the ensemble level. MOVE??
        			  if (newTopology != null) // else this.Pc = this.Pc
        				  topology = UpdateNewTopology(newTopology, W); // line 34: Add the examples during warning to a new topology. TODO: confirm
        			  W.clear();  // line 35  TODO: Move with drift detectors
        			   
        			  // 2.2 Update window size in window properties depending on window size inheritance flag (entry parameter/Option)
        			  this.bkgLearner.windowProperties.setSize(((this.bkgLearner.windowProperties.rememberWindowSize) ? 
        					   this.bkgLearner.internalWindowEvaluator.getWindowSize(this.bkgLearner.indexOriginal) : this.bkgLearner.windowProperties.windowDefaultSize)); 
        		   
        			  // 2.3 Inherit window properties / clear internal evaluator
        			  this.windowProperties = this.bkgLearner.windowProperties; // internalEvaluator shouldnt be inherited
        			  this.internalWindowEvaluator = null; // only a double check, as it should be always null (only used in background + old concept Learners)
        		   }
	            // 2.3 New active classifier is the best retrieved old classifier / clear background learner
                this.classifier = this.bkgLearner.classifier;
                // this.driftDetectionMethod = this.bkgLearner.driftDetectionMethod;
                // this.warningDetectionMethod = this.bkgLearner.warningDetectionMethod;                
                this.evaluator = this.bkgLearner.evaluator;
                this.createdOn = this.bkgLearner.createdOn;		// createdOn = instancesSeen
                this.bkgLearner = null; 
        		} 
            else { 
                this.classifier.resetLearning();
                this.createdOn = instancesSeen;
            }
            this.evaluator.reset();
        }
        
        /** Batch method */
	    public GNG UpdateNewTopology(GNG newTopology, ArrayList<Instance> W){
	    		for(Instance inst: W) {
	    			newTopology.trainOnInstanceImpl(inst);  // TODO: Make sure that we send class as a feature, so everything makes sense after.
	    			
	            	// Should we send instances until we reach the stopping criteria? TODO: Question to acervant               		
	        		/*for(int j=0;newTopology.getCreadas()<newTopology.nParadaOption.getValue();j++){                        	
	        			newTopology.trainOnInstanceImpl(inst);
	                	if(j+1==W.size())
	                		j = -1;
	            	}*/
	    		}
        		return newTopology;  // if topology (Pc) is global, then we don´t need to return this here
        }

        public void trainOnInstance(Instance instance, long instancesSeen) { //	Line 5: (x,y) ← next(S)
            this.classifier.trainOnInstance(instance);    // Line 6: ClassifierTrain(c, x, y) -> Train c on the current instance (x, y).     
        }
	        
        // Creates BKG Classifier in warning window
        public void createBkgClassifier(long lastWarningOn) { // now lastWarningon received as parameter, so it can be passed from an upper level
        		// Empty explicitely the BKG internal evaluator if enabled
        		//if (this.bkgLearner != null) this.bkgLearner.internalWindowEvaluator.clear(); // I don't see any improvement
        	
            // 1 Create a new bkgTree classifier
            Classifier bkgClassifier = this.classifier.copy();
            bkgClassifier.resetLearning();
                                    
            // 2 Resets the evaluator
            BasicClassificationPerformanceEvaluator bkgEvaluator = (BasicClassificationPerformanceEvaluator) this.evaluator.copy();
            bkgEvaluator.reset();
            
            // // System.out.println("------------------------------");
            // // System.out.println("Create estimator for BKG classifier in position: "+this.indexOriginal);
            // 3 Adding also internal evaluator (window) in bkgEvaluator (by @suarezcetrulo)
            DynamicWindowClassificationPerformanceEvaluator bkgInternalWindowEvaluator = null;
            if(this.useRecurringLearner) {
                bkgInternalWindowEvaluator = new DynamicWindowClassificationPerformanceEvaluator (
                		this.windowProperties.getSize(),this.windowProperties.getIncrements(),this.windowProperties.getMinSize(),
                		this.lastError,this.windowProperties.getDecisionThreshold(),true,this.windowProperties.getResizingPolicy(), 
                		this.indexOriginal, "created for BKG classifier in ensembleIndex #"+this.indexOriginal);  
                bkgInternalWindowEvaluator.reset();
            }
            // // System.out.println("------------------------------");
            
            // 4 Create a new bkgLearner object
            this.bkgLearner = new EPCHBaseLearner(indexOriginal, bkgClassifier, bkgEvaluator, 
            				lastWarningOn, // this.lastWarningOn replaced with 'instancesSeen'
            				this.useBkgLearner, this.useDriftDetector, // this.driftDecisionMechanism, 
            				true, this.useRecurringLearner, 
            				false, this.windowProperties, bkgInternalWindowEvaluator, // eventsLogFile, logLevel, 
            				this.warningWindowSizeThreshold); //, new GNG()); // new topology for bkg classifier     	
        }

        public double[] getVotesForInstance(Instance instance) {
            DoubleVector vote = new DoubleVector(this.classifier.getVotesForInstance(instance));
            return vote.getArrayRef();
        }
        
    		// Saves a backup of the active classifier that raised a warning to be stored in the concept history in case of drift.
        public void saveCurrentConcept(long instancesSeen) {  // now instancesSeen is passed by parameter
        		// if(ConceptHistory.historyList != null) // System.out.println("CONCEPT HISTORY SIZE IS: "+ConceptHistory.historyList.size());
        	
        		// 1 Update last error before warning of the active classifier
        		// This error is the total fraction of examples incorrectly classified since this classifier was active until now.
    			this.lastError = this.evaluator.getFractionIncorrectlyClassified();
    			
    			// 2 Copy Base learner for Concept History in case of Drift and store it on temporal object.
    			// First, the internal evaluator will be null. 
    			// It doesn't get initialized till once in the Concept History and the first warning arises. See it in startWarningWindow
    			EPCHBaseLearner tmpConcept = new EPCHBaseLearner(this.indexOriginal, 
    					this.classifier.copy(), (BasicClassificationPerformanceEvaluator) this.evaluator.copy(), 
    					this.createdOn, this.useBkgLearner, this.useDriftDetector, // this.driftDecisionMechanism, 
    					// this.driftOption, this.warningOption, 
    					true, useRecurringLearner, true, this.windowProperties.copy(), null, // eventsLogFile, logLevel, 
    					this.warningWindowSizeThreshold); // , this.topology);
    			// TODO: Save concept in a group

    			// TODO: copy of classifier should be in the actual classifier.
    			this.tmpCopyOfClassifier = new Concept(tmpConcept, 
        				this.createdOn, this.evaluator.getPerformanceMeasurements()[0].getValue(), instancesSeen);
        		
        		// 3 Add the classifier accumulated error (from the start of the classifier) from the iteration before the warning
        		this.tmpCopyOfClassifier.setErrorBeforeWarning(this.lastError);
        		// A simple concept to be stored in the concept history that doesn't have a running learner.
        		// This doesn't train. It keeps the classifier as it was at the beginning of the training window to be stored in case of drift.
        }
       	
    }
       
    
    public static class Group {
    	
    		// params
    		int id;
    		GNG topology;
    		public ConcurrentHashMap<Integer,Concept> groupList; // List of concepts per group
    		
    	    public EPCHBaseLearner copyConcept(int key) {
	    		EPCHBaseLearner aux = groupList.get(key).getBaseLearner();
	    		return aux;
    	    }

		public Group (int id, GNG top) {
			this.id = id; // nextID();
			this.topology = top;
			this.groupList = new ConcurrentHashMap<Integer,Concept>();
		}
		
        /*public int nextID() {
            return ConceptHistory.lastGroupID++;
        }*/
        
        // Getters
        public int getID() {
        		return id;
        }
        
        public ArrayList<double[]> getTopologyPrototypes() {
        		return topology.getPrototypes(); // TODO: IMP. Check that this conversion is accurate (in method of class GNG)
        }
        
    }
    
    /*** 
     * Static and concurrent for all DTs that run in parallel
     * Concept_history = (list of concept_representations)
     */
    public static class ConceptHistory {

    		// Concurrent Concept History List
    		public static ConcurrentHashMap<Integer,Group> historyList; // now this is a list of groups
    		public static int lastID = 0;
    		public static int lastGroupID = 0;
   		
    		// List of ensembles with an active warning used as to determine if the history list evaluators should be in use
    		public static ConcurrentHashMap<Integer,Boolean> classifiersOnWarning; // = new ConcurrentHashMap<Integer,Boolean> ();

	    /*public static EPCHBaseLearner extractConcept(int key) {	// now the concepts are copied and not extracted
	    		EPCHBaseLearner aux = historyList.get(key).getBaseLearner();
	    		historyList.remove(key);
			return aux;
	    }*/
	    
	    public static EPCHBaseLearner copyConcept(int group, int key) {
	    		EPCHBaseLearner aux = historyList.get(group).copyConcept(key);
	    		return aux;
	    }
	    
	    public static int getNumberOfActiveWarnings() {
	    		int count = 0;
	    		for(Boolean value: classifiersOnWarning.values()) if (value) count++;
	    		return count;
	    }
	    
	    /*
		public Set<Entry<Integer,Concept>> getConceptHistoryEntrySet() {
			return historyList.entrySet();
        }
        */
	        
        // Getters
        public static int nextID() {
        		return lastID++;
        }

        public static int nextGroupID() {
        		return lastGroupID++;
        }
    }
    
    //Concept_representation = (classifier, last_weight, last_used_timestamp, conceptual_vector)
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
    		public EPCHBaseLearner ConceptLearner;

    		// Constructor
    		public Concept (EPCHBaseLearner ConceptLearner, long createdOn, double classifiedInstances, long instancesSeen) {
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
	    
	    public EPCHBaseLearner getBaseLearner() {
			return this.ConceptLearner;	    	
	    }   
	    
	    // Setters
	    
	    public void setErrorBeforeWarning(double value) {
	    		this.errorBeforeWarning=value;
	    }
	}
    
    
    // Window-related parameters for classifier internal comparisons during the warning window 
    public class Window {
    	
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
			this.windowDefaultSize=windowSize; // the default size of a window could change overtime if there is window size inheritance enabled
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
		
		public boolean getDynamicWindowInOldClassifiersFlag() {
			return this.backgroundDynamicWindowsFlag;
		}

		public void getDynamicWindowInOldClassifiersFlag(boolean flag) {
			this.backgroundDynamicWindowsFlag = flag;
		}
		
    }

    // Object for events so the code is cleaner
    public class Event {
    	
    		// Fields for events log
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
    		public Event (String [] eventDetails) {
    			 instanceNumber = eventDetails[0];
    			 event = eventDetails[1];
    			 affectedPosition = eventDetails[2];
    			 votingWeigth = eventDetails[3];
    			 warningSetting = eventDetails[4];
    			 driftSetting = eventDetails[5];
    			 createdOn= eventDetails[6];
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
    
    // General auxiliar methods for logging events
    
    public Event getEventHeaders() {
    	
		String [] headers = {
				"instance_number", 
				"event_type", 
				"affected_position", //former 'classifier'
				"voting_weight", // voting weight for the three that presents an event. new 07/07/2018
				"warning_setting", 
				"drift_setting", 
				"affected_classifier_created_on", 
				"error_percentage", 
				"amount_of_classifiers",
				"amount_of_active_warnings",
				"classifiers_on_warning",
				"applicable_concepts",
				"recurring_drift_to_history_id",
				"recurring_drift_to_classifier_created_on"
		};
	
		return (new Event(headers));

}
    
    /**
     * Method to register events such as Warning and Drifts in the event log file.
     * */
	public void logEvent(Event eventDetails) {		
		// Log processed instances, warnings and drifts in file of events
		//# instance, event, affected_position, affected_classifier_id last-error, #classifiers;#active_warnings; classifiers_on_warning, applicable_concepts_from_here, recurring_drift_to_history_id, drift_to_classifier_created_on
		eventsLogFile.println(
				String.join(";",
						eventDetails.getInstanceNumber(), 
						eventDetails.getEvent(), 
						eventDetails.getAffectedPosition(),
						eventDetails.getVotingWeigth(), // of the affected position (this is as represented in getVotesForInstance for the global ensemble), // new 07/07/2018
						eventDetails.getWarningSetting(), // WARNING SETTING of the affected position. new 07/07/2018
						eventDetails.getDriftSetting(), // DRIFT SETTING of the affected position. new 07/07/2018
						eventDetails.getCreatedOn(), // new, affected_classifier_was_created_on
        					eventDetails.getLastError(), 
        					eventDetails.getNumberOfClassifiers(),
        					eventDetails.getNumberOfActiveWarnings(), // #active_warnings
        					eventDetails.getClassifiersOnWarning(), 
        					eventDetails.getListOfApplicableConcepts(), // applicable_concepts_from_here
        					eventDetails.getRecurringDriftToClassifierID(), // recurring_drift_to_history_id
        					eventDetails.getDriftToClassifierCreatedOn())
				);
		eventsLogFile.flush();
		
	}    
        
    
}

package moa.evaluation;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import apple.laf.JRSUIConstants.Size;
import moa.classifiers.meta.RecurringConceptsAdaptiveRandomForest.Concept;
import moa.classifiers.meta.RecurringConceptsAdaptiveRandomForest.ConceptHistory;
import moa.core.Example;
import moa.core.Measurement;
import moa.core.Utils;

import com.github.javacliparser.FlagOption;

/**
 * Classification evaluator that updates evaluation results using a sliding
 * window.
 *
 * @author Andres L. Suarez-Cetrulo (suarezcetrulo at gmail dot com)
 * @version $Revision: 0.1 $
 */
public class DynamicWindowClassificationPerformanceEvaluator extends BasicClassificationPerformanceEvaluator {

    private static final long serialVersionUID = 1L;
	public static int dynamicEstimatorID = 0;

	// First ensemble index pos (when the evaluator was created)
	int indexPos;
	
	// Flags / Options
    public IntOption widthOption = new IntOption("width",'w', "Size of Window", 10);
    public IntOption defaultWidthOption = new IntOption("defaultWidth",'d', "Default Size of Window", 10);
    public IntOption widthIncrementsOption = new IntOption("increments",'i', "Increments in Size of Window", 1);
    public IntOption minWidthOption = new IntOption("minWidth",'m', "Minimum Size of Window", 5);
    public FlagOption backgroundDynamicWindowsFlag = new FlagOption("resizeAllWindows", 'b', "Should the comparison windows for old learners be also dynamic? ");
    public FloatOption thresholdOption = new FloatOption("threshold",'t', "Threshold for resizing", 0.65);
    //public IntOption windowResizePolicyOption = new IntOption("windowResizePolicy",'p', "Policy to update the size of the window. Ordered by complexity, being 0 the simplest one and 3 the one with most complexity.", 0, 0, 2);

	// Window properties
	int windowSize;   
	int defaultSize; // default size when the evaluator was created the first time (the first time its model drifted)
	int windowIncrements;
	int minWindowSize;
	int windowResizePolicy;
	double decisionThreshold;
	double priorEstimation;
	boolean backgroundDynamicWindows;
	boolean useOptions = true; // by default use options
	
	/*HashMap<Integer,Double> estimationBeforeWarning; // REMEMBER ITS 100-accuracy
	HashMap<Integer,Integer> windowSizes; // supporting multiple models to access the same evaluator
	HashMap<Integer,Integer> defaultSizes; // supporting multiple models to access the same evaluator*/
	
	// Variables for methods overrided
    private double totalWeightObserved;
    @SuppressWarnings("unused")
	private int lastSeenClass;


	// Information about classifier that created the estimator
	String createdBy; 

    // Constructor for dynamic internal window evaluators
    public DynamicWindowClassificationPerformanceEvaluator(int windowSize, int windowIncrements, int minWindowSize, 
    		double priorEstimation, double decisionThreshold, boolean resizingEnabled, int windowResizePolicy, int indexPos, String createdBy) {
    	
    		// Initializing and adding values to HashMaps
    		/*this.windowSizes = new HashMap<Integer,Integer>();
    		this.windowSizes.put(indexPos, windowSize);

    		this.defaultSizes = new HashMap<Integer,Integer>();
    		this.defaultSizes.put(indexPos, windowSize);
    		
    		this.estimationBeforeWarning = new HashMap<Integer,Double>();
    		this.estimationBeforeWarning.put(indexPos, priorEstimation);*/
    		
    		this.indexPos=indexPos;
    	
    		// First values
		this.windowSize = windowSize; 
		this.defaultSize =  windowSize;
		this.priorEstimation = priorEstimation;

		this.windowIncrements = windowIncrements;
		this.minWindowSize=minWindowSize;
		this.decisionThreshold=decisionThreshold;
		this.backgroundDynamicWindows=resizingEnabled;
		this.windowResizePolicy=windowResizePolicy;
		this.useOptions=false;
		this.createdBy = createdBy;
    }
    
    public void addModel(int indexPos, double priorEstimation, int windowSize) {
		((DynamicWindowEstimator) this.weightCorrect).addNewModel(indexPos, priorEstimation, windowSize);
    }
    
    public void deleteModel(int indexPos) {
		((DynamicWindowEstimator) this.weightCorrect).deleteModel(indexPos);
    }
    
    public boolean containsIndex(int indexPos) {
    	
    		return ((DynamicWindowEstimator) this.weightCorrect).contains(indexPos);
    }
        
     @Override
    public void reset(){
		this.windowSize =  this.defaultSize;
	    reset(this.numClasses);
	} 
	
    // Overriding next few methods to avoid calculating extra evaluators for kappa statistics - @suarezcetrulo 
    
    @Override
    public void reset(int numClasses) {
        this.numClasses = numClasses;
        this.weightCorrect = newEstimator(); // it will started from scratch for only one window size. 
        // TODO. do we need to improve this to send the whole array of window sizes (from here) if we reset?
        System.out.println("RESET ESTIMATOR ID: "+(dynamicEstimatorID+1));
        this.lastSeenClass = 0;
        this.totalWeightObserved = 0;
    }
    
    // Forcing memory cleaning 
    /*public void emptyEstimator() {
    		((DynamicWindowEstimator) this.weightCorrect).clear();
    		this.weightCorrect = null;
    }*/

    @Override
    public void addResult(Example<Instance> example, double[] classVotes) {
        Instance inst = example.getData();
        double weight = inst.weight();
        if (inst.classIsMissing() == false){
            int trueClass = (int) inst.classValue();
            int predictedClass = Utils.maxIndex(classVotes);
            if (weight > 0.0) {
                if (this.totalWeightObserved == 0) {
                    reset(inst.dataset().numClasses()); // are we using the correct one?
                }
                this.totalWeightObserved += weight;
                this.weightCorrect.add(predictedClass == trueClass ? weight : 0);
            }
            this.lastSeenClass = trueClass;
        }
    }
    
    @Override
    public Measurement[] getPerformanceMeasurements() {
        return new Measurement[]{
                new Measurement("classified instances",
                    getTotalWeightObserved()),
                new Measurement("classifications correct (percent)",
                    getFractionCorrectlyClassified() * 100.0)
        };
    }
    
    public double getFractionIncorrectlyClassified(int ensembleIndex) {
        return 1.0 - getFractionCorrectlyClassified(ensembleIndex) ;
    }
    
    public double getFractionCorrectlyClassified(int ensembleIndex) {
        return ((DynamicWindowEstimator) this.weightCorrect).estimation(ensembleIndex);
    }
    
    // //////////////////////////////////////////////
    
	public void clear(){
		reset(this.numClasses);
	}
	
    // Getters and Setters
    
    /*public void setProperties(int windowSize, int defaultSize, int windowIncrements, int minWindowSize, 
    							 double decisionThreshold, boolean resizingEnabled, int windowResizePolicy) {
		this.windowSize = windowSize;
		this.windowIncrements = windowIncrements;
		this.minWindowSize=minWindowSize;
		this.decisionThreshold=decisionThreshold;
		this.backgroundDynamicWindows=resizingEnabled;
		this.windowResizePolicy=windowResizePolicy;
		this.defaultSize=defaultSize;
		this.useOptions=false;
    }*/
    
    public int getWindowSize(int ensembleIndex) {
	 	return ((DynamicWindowEstimator) weightCorrect).getSize(ensembleIndex);
    }
    
    /*public void setWindowSize(int value) {
	 	this.windowSize=value;
    }*/
    
    
    @Override
    protected Estimator newEstimator() {
    		dynamicEstimatorID=dynamicEstimatorID+1;
    		//System.out.println("NEW WINDOW ESTIMATOR CREATED - call variable weightCorrect for more info.");    		
    		if (useOptions) {
    			// System.out.println("USE OPTIONS");
    			// When using the dynamic estimator as default one, there is no prior estimation (-1), so the only valid resizing policy is 2.
    	        return new DynamicWindowEstimator(this.widthOption.getValue(),this.widthIncrementsOption.getValue(), this.minWidthOption.getValue(), -1, 
    	        									 this.thresholdOption.getValue(), this.backgroundDynamicWindowsFlag.isSet(), 2,
    	        									 DynamicWindowClassificationPerformanceEvaluator.dynamicEstimatorID, 0); // hard coding position in the ensemble = 0
    		}
    		else {  
    			System.out.println("START ESTIMATOR #"+dynamicEstimatorID+" "+this.createdBy);
    	        return new DynamicWindowEstimator(this.windowSize, this.windowIncrements, this.minWindowSize, this.priorEstimation,
	    	        								 this.decisionThreshold,this.backgroundDynamicWindows, this.windowResizePolicy, 
	    	        								 DynamicWindowClassificationPerformanceEvaluator.dynamicEstimatorID, this.indexPos);
    		}		
    }
    
    
    // Estimator

    public class DynamicWindowEstimator implements Estimator {

        /**
		 * by suarezcetrulo
		 */
		 private static final long serialVersionUID = -6525799997539109003L;		 
         public int estimatorID;
        
         // Window container of estimation values
         ArrayList<Double> window;  

         // Dynamic Windows (both will always have the same length)
		 protected HashMap<Integer,Integer> SizeWindows; // AS windowSize
         protected HashMap<Integer,Integer> defaultSizes; 
         
         // Static parameters
         protected int minimumSize; 
         protected int sizeIncrements; 
         protected int windowResizePolicy;
         protected boolean resizingEnabled;
         
         // Resizing decision factor
         protected HashMap<Integer,Double> priorEstimations; // prior error at the beginning of the warning window
         protected double threshold;
         
         // Constructor for window: threshold = -1 means that selected size update policy is 1
         public DynamicWindowEstimator(int initialSize, int sizeIncrements, int minSize, 
        		 double priorEstimation, double threshold, boolean resizingEnabled, 
        		 						  int windowResizePolicy, int estimatorID, int index){
        	 		//System.out.println("Creating estimator #"+estimatorID+" with Window size: "+initialSize);
 	 			//System.out.println("CREATE ESTIMATOR estimator #"+estimatorID+" with Window size: "+initialSize);
      	 		System.out.println("CREATE ESTIMATOR "+estimatorID+" WITH INITIAL POS "+index);

        	 		//System.out.println(index);
        	 		this.estimatorID = estimatorID;
        	 		
        	 		// Initializing dynamic window
    	     		this.window = new ArrayList<Double>();
        	 		
        	 		// Initializing and adding first value in HashMaps
	     		this.SizeWindows= new HashMap<Integer,Integer>();
	     		this.SizeWindows.put(index, initialSize);
	     		
	     		this.defaultSizes = new HashMap<Integer,Integer> ();
	     		this.defaultSizes.put(index, initialSize);

	     		this.priorEstimations = new HashMap<Integer,Double>();
	     		this.priorEstimations.put(index, priorEstimation);
	     		
	     		// Initializing static values
	     		this.sizeIncrements = sizeIncrements;
	         	this.minimumSize = minSize;          // minimum size should always be greater than size increments
	     		this.threshold = threshold;
	     		this.resizingEnabled=resizingEnabled;
	     		this.windowResizePolicy=windowResizePolicy; 
         }
         
         public void addNewModel (int index, double priorEstimation, int windowSize) {
     	 	System.out.println("ADD POS "+index+" IN ESTIMATOR "+estimatorID);
        	 	this.SizeWindows.put(index, windowSize);
        	 	this.defaultSizes.put(index, windowSize);
        	 	this.priorEstimations.put(index, priorEstimation);
         }
         
         public void deleteModel (int index) {
        	 	System.out.println("DELETE POS "+index+" IN ESTIMATOR "+estimatorID);
     	 	this.SizeWindows.remove(index);
     	 	this.defaultSizes.remove(index);
     	 	this.priorEstimations.remove(index);
         }
         
         public boolean contains (int index) {
        		return this.SizeWindows.containsKey(index);
         }

         // It adds errors per classified row
         public void add(double value) {
        	 
        	 	// TEST TRACE
	        /*System.out.println("-------------------------------------------");
	        System.out.println("SIZE: "+SizeWindows.size());
	        	for (Entry<Integer, Integer> modelWindows : SizeWindows.entrySet()) {
	        	 	System.out.println("pos: "+modelWindows.getKey()+" in estimator #"+this.estimatorID+":  Adding result: "+value+" - window size was:"+modelWindows.getValue());
	         	System.out.println("pos: "+modelWindows.getKey()+" actual window is: "+getSublist(modelWindows.getValue())); //, this.sizeIncrements));
	         	System.out.println("pos: "+modelWindows.getKey()+" small window is:"+getSublist(modelWindows.getValue()-this.sizeIncrements)); //, this.sizeIncrements*2));
	        	}
        	 	System.out.println("---");
	        System.out.println("Complete-largest window is: "+this.window);
     	 	System.out.println("---");*/
     	 	// END TEST TRACE
     	 	
  			// Remove oldest if it surpasses the maximum windowSize + increments
         	if(window.size()>=(Collections.max(this.SizeWindows.values())+this.sizeIncrements)) this.window.remove(0);
         		         	
        	 	// Always storing extra results just in case the window grows
  			this.window.add(value);
	  		
	        	for (Entry<Integer, Integer> modelWindows : SizeWindows.entrySet()) {
	         	// Resize window -- this update window size here is designed for increments of 1 unit, 
	  			// as it is executed every time one instance is created. 
	  			// Otherwise, the window could grow faster than the incoming data and be pointless. 
	  			// The single remove statement above also follows this.
	         	if (this.resizingEnabled) updateWindowSize(modelWindows.getKey()); 
	         	
	  			// TEST TRACE
	         	//System.out.println("// // // // (after resizing)");
	         	//System.out.println("estimatorID: "+estimatorID+" pos: "+modelWindows.getKey()+" window size is:"+modelWindows.getValue());
	         	/* System.out.println("pos: "+modelWindows.getKey()+" actual window is: "+getSublist(modelWindows.getValue())); //, this.sizeIncrements));
	         	System.out.println("pos: "+modelWindows.getKey()+" small window is:"+getSublist(modelWindows.getValue()-this.sizeIncrements)); //, this.sizeIncrements*2));
	         	System.out.println(); */
	        	}
        	 	/*System.out.println("---");
        	 	System.out.println("Complete-largest window is: "+this.window);
     	 	System.out.println("---");
         	System.out.println("-------------------------------------------");*/
     	 	// END TEST TRACE

         }
         
         // Returns the latest error window creating a sublist of the last errors.
         // As we store the plus last INCREMENT_SIZE errors, we need to start from the position INCREMENT_SIZE-1 (zero indexed)
         // Size of the sublist is WINDOW_SIZE 
         public double estimation(int pos){ // = getLastWindowEstimation
    	 		//if (this.window.size() == 0) return getLargeWindowEstimation ();
    	 		//else {
        	 		// Parameters for sublists
        	 		
        	 		// Print estimations 
        	 		/*System.out.println("In Estimator #"+this.estimatorID+" with #"+this.window.size()+" examples in total and WindowSize is "+SizeWindow+" so prediction is :"
        	 				+(estimateError(this.window.subList((this.window.size() < this.SizeWindow ? 
	 								0 : (this.window.size() - generalStartPoint) < this.SizeWindow ? 
	 								specialStartPoint : generalStartPoint 
			        	 		), this.window.size()))));*/
        	 		//System.out.println("Error for estimator #"+this.estimatorID+": "+estimateError(getEstimationSublist(this.SizeWindow, this.sizeIncrements)));
        	 		//System.out.println(this.window.toString());
        	 
        	 		return estimateError(getSublist(this.SizeWindows.get(pos)));//, this.sizeIncrements));
    	 		//}
         }
         
         // gets estimation for only first element (default, it will work in bkg evaluators - but not old concepts)
         public double estimation(){
        	 	System.out.println("WARNING!!!!!! USING WRONG ESTIMATOR!!");
 	 		return estimateError(getSublist(this.SizeWindows.get(0))); //, this.sizeIncrements));
 	 		// TODO: DO WE ACTUALLY NEED THE DEFAULT ESTIMATOR? IS IT USED?
        	 }

         public double getSmallerWindowEstimation (int pos){
        	 	return estimateError(getSublist(this.SizeWindows.get(pos)-this.sizeIncrements));//, this.sizeIncrements*2));
         }
         
         public double getLargeWindowEstimation (int pos){
     			//return estimateError(this.window); // full window (with +1 increment)
        	 		return estimateError(getSublist(this.SizeWindows.get(pos)+this.sizeIncrements));
         }

         
 		/** EXPLANATION by @author suarezcetrulo:
 		 *  -End of sublist: we always finish in this.actualSize (as it's where the latest results are)
 		 *  -Start of sublist there are three different cases:
 		 * 		+ If actualSize < desiredWindowSize 
 		 * 		  THEN we start from pos: 0 (as we need the whole list to satisfy as much as we can the desiredSize)
 		 * 		+ Else if (actualSize - startPoint) < desiredWindowSize 
 		 * 		  THEN we start from pos: actualSize % desiredWindowSize (this will give us the residual that doesn't fit in our desiredSize and we can use as startPoint)
 		 * 		+ Otherwise: we start from the general startPoint (which is = largestWindowSize-desiredWindowSize)
 		 * 
 		 * @param desiredSize: window size or smaller window size
 		 * @param generalStartPoint: this.sizeIncrements or this.sizeIncrements*2 (for smallerWindows)
 		 * @return sublist with the desired sub window
 		 */
         public List<Double> getSublist(int desiredSize){ //, int generalStartPoint){
	        	 /*return this.window.subList((this.window.size() < desiredSize ? 
							0 : (this.window.size() - generalStartPoint) < desiredSize ? 
							this.window.size() % desiredSize : generalStartPoint 
	     	 		), this.window.size()); */ // this was only good for a single dynamic window and not for multiple as now with the HashMaps	        	 
	        	 return this.window.subList(Math.max(this.window.size() - desiredSize, 0), this.window.size());
         }
                 
         public double estimateError(List<Double> list) {
         	  Double sum = 0.0;
         	  if(!list.isEmpty()) {
         	    for (Double error : list) {
         	        sum += error;
         	    }
         	    //System.out.println("Error is: "+(sum.doubleValue() / (double) list.size()));
         	    return sum.doubleValue() / (double) list.size();
         	  }
         	  return sum;         	  
         }
        
         public void updateWindowSize(int pos) {
        	 
    	 		switch(this.windowResizePolicy) {
    	 		
        	 		case 0: // TODO -> CHECK THIS
             		// size update policy 0 [ min(error(c,w_c)) - priorError ] >= 0 -> window size increases
             		//if(priorError!=-1) { .. }
        	 			// working with accuracy results
        	 			// TEST TRACES
        	 			//System.out.println("estimatorID: "+this.estimatorID+" pos:"+pos+" currentEstimation is: "+estimation(pos));
        	 			//System.out.println("estimatorID: "+this.estimatorID+" pos:"+pos+" active Model Estimation until Warning was: "+this.priorEstimations.get(pos));
         			if(estimation(pos) < this.priorEstimations.get(pos))
         				this.SizeWindows.put(pos,this.SizeWindows.get(pos)+this.sizeIncrements);
         			else // otherwise it decreases
     					this.SizeWindows.put(pos,this.SizeWindows.get(pos)-this.sizeIncrements);
         				if(this.SizeWindows.get(pos)<=this.minimumSize) 
         					this.SizeWindows.put(pos,this.minimumSize);
         			break;
         			
        	 		case 1:
             		// size update policy 1 [ min(error(c,w_c)) - priorError ] > threshold -> window size increases
             		//if(priorError!=-1) { .. }
         			if(estimation(pos) - this.priorEstimations.get(pos) <= this.threshold)
         				this.SizeWindows.put(pos,this.SizeWindows.get(pos)+this.sizeIncrements);
         			else // otherwise it decreases
     					this.SizeWindows.put(pos,this.SizeWindows.get(pos)-this.sizeIncrements);
         				if(this.SizeWindows.get(pos)<this.minimumSize) 
         					this.SizeWindows.put(pos,this.minimumSize);
         			break;
         			
        	 		case 2:
        	 			//For each c, W_c=s  , where s is an small number / For each iteration, independently of ny situations / a = active model
        	 			//if(priorError==-1) { .. }
        	 			// -100 to work with error
        	 			// working with accuracy results
             		double W_a_candidate_0=100-estimation(pos); // or  W_a_candidate_0=getAverageOfErrors(getLastWindow())
             		double W_a_candidate_1=100-getLargeWindowEstimation(pos);
             		double W_a_candidate_2=100-getSmallerWindowEstimation (pos);
        		
             		if ( W_a_candidate_1 > W_a_candidate_0 && W_a_candidate_1 > W_a_candidate_2 ) // Increase window size
         				this.SizeWindows.put(pos,this.SizeWindows.get(pos)+this.sizeIncrements);
             		else if ( W_a_candidate_2 > W_a_candidate_0 && W_a_candidate_2 > W_a_candidate_1 ) // Decrease window size
     					this.SizeWindows.put(pos,this.SizeWindows.get(pos)-this.sizeIncrements);
         				if(this.SizeWindows.get(pos)<this.minimumSize) 
         					this.SizeWindows.put(pos,this.minimumSize);
                   //else System.out.println("Window size remains the same");
        	 			break;
        	 		  // no default case
    	 		}		
         }
         
         public void reset(int pos) {
         		this.window.clear();
         		this.SizeWindows.put(pos,defaultSize);
         }
         
         // Clean full window
         /* public void clear() {
         		this.window.clear();
         }*/
         
         public int getSize(int pos) {
        	 	return this.SizeWindows.get(pos);
         }

    }

}


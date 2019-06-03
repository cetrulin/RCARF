package moa.evaluation;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

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
    protected String evaluatorType;  // optional (for EPCH: ACTIVE, BKG or CH)


	// Variables for methods overrided
    private double totalWeightObserved;
    @SuppressWarnings("unused")
	private int lastSeenClass;

	// Information about classifier that created the estimator
	String createdBy; 

    // Constructor for dynamic internal window evaluators
    public DynamicWindowClassificationPerformanceEvaluator(int windowSize, int windowIncrements, int minWindowSize, 
    		double priorEstimation, double decisionThreshold, boolean resizingEnabled, int windowResizePolicy, int indexPos, String createdBy) {
    	
    		// Initializing 
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
    
    public int getAmountOfApplicableModels() {
	 	return ((DynamicWindowEstimator) this.weightCorrect).getAmountOfApplicableModels();
    }
    
	/*public void clear(){
		((DynamicWindowEstimator) this.weightCorrect).clear();
		this.weightCorrect = null;
		// reset(this.numClasses);
	}*/
	
    // Getters and Setters
    
    public int getWindowSize(int ensembleIndex) {
	 	return ((DynamicWindowEstimator) weightCorrect).getSize(ensembleIndex);
    }
    
    // Overriding next few methods to avoid calculating extra evaluators for kappa statistics - @suarezcetrulo 
    @Override
    public void reset(){
		this.windowSize =  this.defaultSize;
	    reset(this.numClasses);
	} 
	    
    @Override
    public void reset(int numClasses) {
        this.numClasses = numClasses;
        this.weightCorrect = newEstimator(); // it will started from scratch for only one window size. 
        // System.out.println("RESET ESTIMATOR ID: "+(dynamicEstimatorID+1)+" "+createdBy+" "+" pos:"+this.indexPos);
        this.lastSeenClass = 0;
        this.totalWeightObserved = 0;
    }

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
    
    // The index sent belongs to the active model that the evaluator compares against (it should be an applicable model)
    public double getFractionIncorrectlyClassified(int ensembleIndex) {
        return 1.0 - getFractionCorrectlyClassified(ensembleIndex) ;
    }
    
    // The index sent belongs to the active model  against (it should be an applicable model)
    public double getFractionCorrectlyClassified(int ensembleIndex) {
        return ((DynamicWindowEstimator) this.weightCorrect).estimation(ensembleIndex);
    }
    
    // //////////////////////////////////////////////


    @Override
    protected Estimator newEstimator() {
    		dynamicEstimatorID=dynamicEstimatorID+1;
    		//System.out.println("NEW WINDOW ESTIMATOR CREATED - call variable weightCorrect for more info.");    		
    		if (useOptions) {
    			// When using the dynamic estimator as default one, there is no prior estimation (-1), so the only valid resizing policy is 2.
    	        return new DynamicWindowEstimator(this.widthOption.getValue(),this.widthIncrementsOption.getValue(), this.minWidthOption.getValue(), -1, 
    	        									 this.thresholdOption.getValue(), this.backgroundDynamicWindowsFlag.isSet(), 2,
    	        									 DynamicWindowClassificationPerformanceEvaluator.dynamicEstimatorID, 0); // hard coding position in the ensemble = 0
    		}
    		else {  
    			System.out.println();System.out.println();System.out.println();
    			System.out.println("NEW "+this.evaluatorType+" ESTIMATOR WITH ID: "+DynamicWindowClassificationPerformanceEvaluator.dynamicEstimatorID);
    			System.out.println();System.out.println();System.out.println();
    			//System.out.println("START ESTIMATOR #"+dynamicEstimatorID+" "+this.createdBy);
    	        return new DynamicWindowEstimator(this.windowSize, this.windowIncrements, this.minWindowSize, this.priorEstimation,
	    	        								 this.decisionThreshold,this.backgroundDynamicWindows, this.windowResizePolicy, 
	    	        								 DynamicWindowClassificationPerformanceEvaluator.dynamicEstimatorID, this.indexPos);
    		}		
    }
    
    
    public String getEvaluatorType() {
		return evaluatorType;
	}

	public void setEvaluatorType(String evaluatorType) {
		this.evaluatorType = evaluatorType;
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
      	 		//System.out.println("CREATE ESTIMATOR "+estimatorID+" WITH INITIAL POS "+index);

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
         
         public int getAmountOfApplicableModels() {
        	 	return this.SizeWindows.size();
         }
         
         public void addNewModel (int index, double priorEstimation, int windowSize) {
     	 	//System.out.println("ADD APPLICABLE MODEL "+index+" IN ESTIMATOR "+estimatorID);
        	 	this.SizeWindows.put(index, windowSize);
        	 	this.defaultSizes.put(index, windowSize);
        	 	this.priorEstimations.put(index, priorEstimation);
         }
         
         public void deleteModel (int index) {
        	 	//System.out.println("DELETE APPLICABLE MODEL "+index+" IN ESTIMATOR "+estimatorID);
     	 	this.SizeWindows.remove(index);
     	 	this.defaultSizes.remove(index);
     	 	this.priorEstimations.remove(index);
         }
         
         public boolean contains (int index) {
        		return this.SizeWindows.containsKey(index);
         }
         
         public int getSize(int pos) {
        	 	return this.SizeWindows.get(pos);
         }
         
         // It adds errors per classified row
         public void add(double value) {
        	    
        	 	// TEST TRACE
	        /*System.out.println("-------------------------------------------");
	        System.out.println("SIZE: "+SizeWindows.size());*/
        	 
	        	for (Entry<Integer, Integer> modelWindows : SizeWindows.entrySet()) {
	        		System.out.println("ESTIMATOR: "+this.estimatorID);
	        	 	System.out.println("pos: "+modelWindows.getKey()+" in estimator #"+this.estimatorID+":  Adding result: "+value+" - window size should be:"+modelWindows.getValue());
	         	System.out.println("pos: "+modelWindows.getKey()+" actual window is: "+getSublist(modelWindows.getValue())); //, this.sizeIncrements));
	         	//System.out.println("pos: "+modelWindows.getKey()+" small window is:"+getSublist(modelWindows.getValue()-this.sizeIncrements)); //, this.sizeIncrements*2));
	        	}
        	 	/*System.out.println("---");
	        System.out.println("Complete-largest window is: "+this.window);
     	 	System.out.println("---"); 	 	
        	 	System.out.println("-- WINDOWS SIZE IS: "+window.size()+" THERE ARE "+SizeWindows.size()+" APPLICABLE MODELS");
     	 	System.out.println("-- APPLICABLE MODELS ARE: "+SizeWindows.keySet()+" AND THEIR SIZES ARE "+SizeWindows.values());
     	 	System.out.println("-- SO MAX WINDOW SIZE IS: "+Collections.max(this.SizeWindows.values())+" + WINDOWS_INCREMENT = FULL WINDOW(?)");
      	 	// END TEST TRACE*/

        	 	// Always storing extra results just in case the window grows
			this.window.add(value);
			
  			// Remove oldest if it surpasses the maximum windowSize + increments. Also allow it to grow at the start till the minimum size if the default size is lower than this.
         	if(this.window.size()>=(Math.max((Collections.max(this.SizeWindows.values())+this.sizeIncrements), this.minimumSize))) 
         		this.window.remove(0);
	  		
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
         // Size of the sublist is WINDOW_SIZE 
         public double estimation(int pos){ // = getLastWindowEstimation        	 
        	 		return estimateError(getSublist(this.SizeWindows.get(pos)));
         }
         
         // DEFAULT ESTIMATOR
         public double estimation(){
        	 	System.out.println("WARNING!!!!!! USING WRONG ESTIMATOR!!");
 	 		return estimateError(getSublist(this.SizeWindows.get(0))); 
        	 }

         public double getSmallerWindowEstimation (int pos){ 
        	 // TODO: this reduces the window to minimum size - 1 (for sizeIncrementes=1).
        	 // this only affects to the third policy (policy value = 2).
        	 // tests can be run anyway, just bearing on mind that the smallerWindow wont respect this size (dont set minimum at 1 if selecting third policy)
        	 	return estimateError(getSublist(this.SizeWindows.get(pos)-this.sizeIncrements));
         }
         
         public double getLargeWindowEstimation (int pos){
        	 		return estimateError(getSublist(this.SizeWindows.get(pos)+this.sizeIncrements));
         }

         
 		/**
 		 * @param desiredSize: window size or smaller window size
 		 * @return sublist with the desired sub window
 		 */
         public List<Double> getSublist(int desiredSize){       	 
	        	 return this.window.subList(Math.max(this.window.size() - desiredSize, 0), this.window.size());
         }
                 
         public double estimateError(List<Double> list) {
        	 	Double sum = 0.0;
        	 	if(!list.isEmpty()) {
				for (Double error : list) {
					sum += error;
				} return sum.doubleValue() / (double) list.size();
			} return sum;         	  
         }
        
         public void updateWindowSize(int pos) {
        	 
    	 		switch(this.windowResizePolicy) {
    	 		
        	 		case 0: // working with ERROR 
             		// size update policy 0 [ min(error(c,w_c)) - priorError ] >= 0 -> window size increases
             		//if(priorError!=-1) { .. }
        	 			// working with ERRROR 
        	 			// TEST TRACES
        	 			System.out.println("estimatorID: "+this.estimatorID+" pos:"+pos+" currentEstimation is: "+estimation(pos));
        	 			//System.out.println("estimatorID: "+this.estimatorID+" pos:"+pos+" active Model Estimation until Warning was: "+this.priorEstimations.get(pos));
         			if(estimation(pos) < this.priorEstimations.get(pos))
         				this.SizeWindows.put(pos,this.SizeWindows.get(pos)+this.sizeIncrements);
         			else { // otherwise it decreases
     					this.SizeWindows.put(pos,this.SizeWindows.get(pos)-this.sizeIncrements);
     					System.out.println("MINIMUM SIZE IS: "+this.minimumSize+" AND ACTUAL SIZE IS: "+getSublist(this.SizeWindows.get(pos)).size()); 
         				if(getSublist(this.SizeWindows.get(pos)).size()<this.minimumSize) this.SizeWindows.put(pos,this.minimumSize);
         				else { //if window size is greater than minimum size, the evaluator is decreasing size and its the maximum size model, we delete a value in the window
         					System.out.println("ENTERED AS ACTUAL SIZE IS: "+this.SizeWindows.get(pos)+" ALTHOUGH ACTUAL SIZE IS: "+getSublist(this.SizeWindows.get(pos)).size());
         					if (this.SizeWindows.get(pos) == Collections.max(this.SizeWindows.values())){
         						for(int n = 1; n <= this.sizeIncrements ; n++) this.window.remove(0); // we do it as many times as values we insert per time (sizeIncrements)
         					}
         				}
         			}
         			break;
         		// TODO: POLICIES 1 AND 2 STILL NEED TESTING	
        	 		case 1:  // working with ERRROR 
             		// size update policy 1 [ min(error(c,w_c)) - priorError ] > threshold -> window size increases
             		//if(priorError!=-1) { .. }
         			if(estimation(pos) - this.priorEstimations.get(pos) <= this.threshold)
         				this.SizeWindows.put(pos,this.SizeWindows.get(pos)+this.sizeIncrements);
         			else {// otherwise it decreases
     					this.SizeWindows.put(pos,this.SizeWindows.get(pos)-this.sizeIncrements);
         				if(getSublist(this.SizeWindows.get(pos)).size()<this.minimumSize) 
         					this.SizeWindows.put(pos,this.minimumSize);
         				else { //if window size is greater than minimum size, it is decreasing size and its the maximum size model, we delete a value in the window to free space
         					if (this.SizeWindows.get(pos) == Collections.max(this.SizeWindows.values())){
         						for(int n = 1; n <= this.sizeIncrements ; n++) this.window.remove(0); // we do it as many times as values we insert per time (sizeIncrements)
         					}
         				}
         			}
         			break;
         			
        	 		case 2:  // working with ACCURACY 
        	 			//For each c, W_c=s  , where s is an small number / For each iteration, independently of ny situations / a = active model
        	 			//if(priorError==-1) { .. }
             		double W_a_candidate_0=100-estimation(pos); // or  W_a_candidate_0=getAverageOfErrors(getLastWindow())
             		double W_a_candidate_1=100-getLargeWindowEstimation(pos);
             		double W_a_candidate_2=100-getSmallerWindowEstimation (pos);
        		
             		if ( W_a_candidate_1 > W_a_candidate_0 && W_a_candidate_1 > W_a_candidate_2 ) // Increase window size
         				this.SizeWindows.put(pos,this.SizeWindows.get(pos)+this.sizeIncrements);
             		else if ( W_a_candidate_2 > W_a_candidate_0 && W_a_candidate_2 > W_a_candidate_1 ) { // Decrease window size
     					this.SizeWindows.put(pos,this.SizeWindows.get(pos)-this.sizeIncrements);
         				if(getSublist(this.SizeWindows.get(pos)).size()<this.minimumSize) 
         					this.SizeWindows.put(pos,this.minimumSize);
         				else { //if window size is greater than minimum size, it is decreasing size and its the maximum size model, we delete a value in the window
         					if (this.SizeWindows.get(pos) == Collections.max(this.SizeWindows.values())){
         						for(int n = 1; n <= this.sizeIncrements ; n++) this.window.remove(0); // we do it as many times as values we insert per time (sizeIncrements)
         					}
         				}
             		}
        	 			break;
    	 		}		
         }
         
         /*public void reset(int pos) {
         		this.window.clear();
         		this.SizeWindows.put(pos,defaultSize);
         }*/
         
         // Clean full window
         /*public void clear() {
		 	 	this.SizeWindows.clear();
		 	 	this.SizeWindows = null;
		  	this.defaultSizes.clear();
			 	this.defaultSizes = null;
			 	this.priorEstimations.clear();
			 	this.priorEstimations = null;
			 	this.window.clear();
			 	this.window = null;
		  }*/
         
    }

}


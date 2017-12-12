package moa.evaluation;
import java.util.ArrayList;
import java.util.List;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.core.Example;
import moa.core.Measurement;
import moa.core.Utils;
import moa.evaluation.BasicClassificationPerformanceEvaluator.Estimator;

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
	int defaultSize;
	int windowIncrements;
	int minWindowSize;
	int windowResizePolicy;
	double decisionThreshold;
	double priorEstimation;
	boolean backgroundDynamicWindows;
	boolean useOptions = true; // by default use options
	
	// Variables for methods overrided
    private double totalWeightObserved;
    @SuppressWarnings("unused")
	private int lastSeenClass;


	// Information about classifier that created the estimator
	String createdBy; 
    
    // Specific constructor for Basic Classification Performance Evaluator (measures overall accuracy of the ensemble)
    /*public DynamicWindowClassificationPerformanceEvaluator(boolean staticWindow) {
    		if (staticWindow) {
	    		this.windowSize = -1;
	    		this.defaultSize =  -1;
	    		this.windowIncrements = 0;
	    		this.minWindowSize=-1;
	    		this.decisionThreshold=-1.0;
	    		this.backgroundDynamicWindows=false;
	    		this.useOptions=false;
    		} // else, dynamic window with option/parameters values
    }*/
    
    // Constructor for dynamic internal window evaluators
    public DynamicWindowClassificationPerformanceEvaluator(int windowSize, int windowIncrements, int minWindowSize, 
    		double priorEstimation, double decisionThreshold, boolean resizingEnabled, int windowResizePolicy, String createdBy) {
    	
		this.windowSize = windowSize; 
		this.defaultSize =  windowSize;
		this.windowIncrements = windowIncrements;
		this.minWindowSize=minWindowSize;
		this.decisionThreshold=decisionThreshold;
		this.backgroundDynamicWindows=resizingEnabled;
		this.windowResizePolicy=windowResizePolicy;
		this.priorEstimation = priorEstimation;
		this.useOptions=false;
		this.createdBy = createdBy;
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
        this.weightCorrect = newEstimator();
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
                    reset(inst.dataset().numClasses());
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
    
    // //////////////////////////////////////////////
    
	public void clear(){
		reset(this.numClasses);
	}
	
    // Getters and Setters
    
    public void setProperties(int windowSize, int defaultSize, int windowIncrements, int minWindowSize, 
    							 double decisionThreshold, boolean resizingEnabled, int windowResizePolicy) {
		this.windowSize = windowSize;
		this.windowIncrements = windowIncrements;
		this.minWindowSize=minWindowSize;
		this.decisionThreshold=decisionThreshold;
		this.backgroundDynamicWindows=resizingEnabled;
		this.windowResizePolicy=windowResizePolicy;
		this.defaultSize=defaultSize;
		this.useOptions=false;
    }
    
    public int getWindowSize() {
	 	return ((DynamicWindowEstimator) weightCorrect).getSize();
    }
    
    public void setWindowSize(int value) {
	 	this.windowSize=value;
    }
    
    
    @Override
    protected Estimator newEstimator() {
    		dynamicEstimatorID=dynamicEstimatorID+1;
    		//System.out.println("NEW WINDOW ESTIMATOR CREATED - call variable weightCorrect for more info.");    		
    		if (useOptions) {
    			// System.out.println("USE OPTIONS");
    			// When using the dynamic estimator as default one, there is no prior estimation (-1), so the only valid resizing policy is 2.
    	        return new DynamicWindowEstimator(this.widthOption.getValue(),this.widthIncrementsOption.getValue(), this.minWidthOption.getValue(), -1, 
    	        									 this.thresholdOption.getValue(), this.backgroundDynamicWindowsFlag.isSet(), 2,
    	        									 DynamicWindowClassificationPerformanceEvaluator.dynamicEstimatorID);
    		}
    		else {  
    			//System.out.println("START ESTIMATOR #"+dynamicEstimatorID+" "+this.createdBy);
    	        return new DynamicWindowEstimator(this.windowSize, this.windowIncrements, this.minWindowSize, this.priorEstimation,
	    	        								 this.decisionThreshold,this.backgroundDynamicWindows, this.windowResizePolicy, 
	    	        								 DynamicWindowClassificationPerformanceEvaluator.dynamicEstimatorID);
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

         // For static windows
         protected double len;
         protected double sum;
		 
         // Dynamic Windows
		 protected int SizeWindow; // AS windowSize
         protected int minimumSize; 
         protected int defaultSize; 
         protected int sizeIncrements; 
         protected int windowResizePolicy;
         protected boolean resizingEnabled;
         
         // Resizing decision factor
         protected double priorEstimation; // prior error at the beginning of the warning window
         protected double threshold;
         
         // Constructor for window: threshold = -1 means that selected size update policy is 1
         public DynamicWindowEstimator(int initialSize, int sizeIncrements, int minSize, 
        		 						  double priorEstimation, double threshold, boolean resizingEnabled, 
        		 						  int windowResizePolicy, int estimatorID){
        	 		//System.out.println("Creating estimator #"+estimatorID+" with Window size: "+initialSize);
        	 		this.estimatorID = estimatorID;
	     		this.SizeWindow = initialSize;
	     		this.defaultSize = initialSize;
	     		this.sizeIncrements = sizeIncrements;
	         	this.minimumSize = minSize;
	     		this.threshold = threshold;
	     		this.window = new ArrayList<Double>();
	     		this.resizingEnabled=resizingEnabled;
	     		this.priorEstimation=priorEstimation;
	     		this.windowResizePolicy=windowResizePolicy; 
         }

         // It adds errors per classified row
         public void add(double value) {
        	 	//System.out.println("Dynamic Window --> Adding result: "+value+" - window size was:"+this.SizeWindow);
  			// Remove oldest if it surpasses windowSize + increments
         	if(window.size()>=(this.SizeWindow+this.sizeIncrements)) this.window.remove(0);

        	 	// Always storing extra results just in case the window grows
  			this.window.add(value);
  			
         	// Resize window -- this update window size here is designed for increments of 1 unit, 
  			// as it is executed everytime one instance is created. 
  			// Otherwise, the window could grow faster than the incoming data and be pointless
         	if (this.resizingEnabled) updateWindowSize(); 
         	//System.out.println("After resizing, window size is:"+this.SizeWindow);
         }
         
         // Returns the latest error window creating a sublist of the last errors.
         // As we store the plus last INCREMENT_SIZE errors, we need to start from the position INCREMENT_SIZE-1 (zero indexed)
         // Size of the sublist is WINDOW_SIZE 
         public double estimation(){ // = getLastWindowEstimation
    	 		//if (this.window.size() == 0) return getLargeWindowEstimation ();
    	 		//else {
        	 		// Parameters for sublists
        	 		int generalStartPoint = this.sizeIncrements; // we always have one extra increment in the whole window (when full)  
        	 		
        	 		// Print estimations 
        	 		/*System.out.println("In Estimator #"+this.estimatorID+" with #"+this.window.size()+" examples in total and WindowSize is "+SizeWindow+" so prediction is :"
        	 				+(estimateError(this.window.subList((this.window.size() < this.SizeWindow ? 
	 								0 : (this.window.size() - generalStartPoint) < this.SizeWindow ? 
	 								specialStartPoint : generalStartPoint 
			        	 		), this.window.size()))));*/
        	 		return estimateError(getEstimationSublist(this.SizeWindow,generalStartPoint));
    	 		//}
         }

         
         public double getSmallerWindowEstimation (){
    	 		// Parameters for sublists
    	 		int smallerSize = this.SizeWindow-this.sizeIncrements;
    	 		int generalStartPoint = this.sizeIncrements*2; // one extra decrement
    	 		return estimateError(getEstimationSublist(smallerSize,generalStartPoint));
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
         public List<Double> getEstimationSublist(int desiredSize, int generalStartPoint){
	        	 return this.window.subList((this.window.size() < desiredSize ? 
							0 : (this.window.size() - generalStartPoint) < desiredSize ? 
							this.window.size() % desiredSize : generalStartPoint 
	     	 		), this.window.size()); 
         }
         
         public double getLargeWindowEstimation (){
     			return estimateError(this.window); // full window (with +1 increment)
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
        
         public void updateWindowSize() {
        	 
    	 		switch(this.windowResizePolicy) {
    	 		
        	 		case 0: // TODO -> CHECK THIS
             		// size update policy 0 [ min(error(c,w_c)) - priorError ] >= 0 -> window size increases
             		//if(priorError!=-1) { .. }
         			if(estimation() >= this.priorEstimation)
         				this.SizeWindow+=this.sizeIncrements;
         			else // otherwise it decreases
         				this.SizeWindow-=this.sizeIncrements;
         				if(this.SizeWindow<=this.minimumSize) 
         					this.SizeWindow=this.minimumSize;
         			break;
         			
        	 		case 1:
             		// size update policy 1 [ min(error(c,w_c)) - priorError ] > threshold -> window size increases
             		//if(priorError!=-1) { .. }
         			if(estimation() - this.priorEstimation > this.threshold)
         				this.SizeWindow+=this.sizeIncrements;
         			else // otherwise it decreases
         				this.SizeWindow-=this.sizeIncrements;
         				if(this.SizeWindow<this.minimumSize) 
         					this.SizeWindow=this.minimumSize;
         			break;
         			
        	 		case 2:
        	 			//For each c, W_c=s  , where s is an small number / For each iteration, independently of ny situations / a = active model
        	 			//if(priorError==-1) { .. }
             		double W_a_candidate_0=estimation(); // or  W_a_candidate_0=getAverageOfErrors(getLastWindow())
             		double W_a_candidate_1=getLargeWindowEstimation();
             		double W_a_candidate_2=getSmallerWindowEstimation ();
        		
             		if ( W_a_candidate_1 > W_a_candidate_0 && W_a_candidate_1 > W_a_candidate_2 ) // Increase window size
         				this.SizeWindow+=this.sizeIncrements;
             		else if ( W_a_candidate_2 > W_a_candidate_0 && W_a_candidate_2 > W_a_candidate_1 ) // Decrease window size
         				this.SizeWindow-=this.sizeIncrements;
         				if(this.SizeWindow<this.minimumSize) 
         					this.SizeWindow=this.minimumSize;
                   //else System.out.println("Window size remains the same");
        	 			break;
        	 		  // no default case
    	 		}		
         }
         
         
         public void reset() {
         		this.window.clear();
         		this.SizeWindow=defaultSize;
         }
         
         public void clear() {
         		this.window.clear();
         }
         
         public int getSize() {
        	 	return this.SizeWindow;
         }

    }

}


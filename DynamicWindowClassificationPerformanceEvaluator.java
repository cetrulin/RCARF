package moa.evaluation;
import java.util.ArrayList;
import java.util.List;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
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
    public DynamicWindowClassificationPerformanceEvaluator(int windowSize, int windowIncrements, int minWindowSize, double priorEstimation, double decisionThreshold, boolean resizingEnabled, int windowResizePolicy) {
		this.windowSize = windowSize; // == windowSize
		this.defaultSize =  windowSize;
		this.windowIncrements = windowIncrements;
		this.minWindowSize=minWindowSize;
		this.decisionThreshold=decisionThreshold;
		this.backgroundDynamicWindows=resizingEnabled;
		this.windowResizePolicy=windowResizePolicy;
		this.priorEstimation = priorEstimation;
		// default size?
		this.useOptions=false;
    }
    
    @Override
    public void reset(){
		this.windowSize =  this.defaultSize;
	    reset(this.numClasses);
	}
	
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
    			System.out.println("USE OPTIONS");
    			// When using the dynamic estimator as default one, there is no prior estimation (-1), so the only valid resizing policy is 2.
    	        return new DynamicWindowEstimator(this.widthOption.getValue(),this.widthIncrementsOption.getValue(), this.minWidthOption.getValue(), -1, 
    	        									 this.thresholdOption.getValue(), this.backgroundDynamicWindowsFlag.isSet(), 2,
    	        									 DynamicWindowClassificationPerformanceEvaluator.dynamicEstimatorID);
    		}
    		else {  
    			System.out.println("START ESTIMATOR #"+dynamicEstimatorID+"   -   of SIZE: "+this.windowSize);
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
         protected double priorEstimation; // error at the beginning of the warning window
         protected double threshold;
         
         // Constructor for window: threshold = -1 means that selected size update policy is 1
         public DynamicWindowEstimator(int initialSize, int sizeIncrements, int minSize, 
        		 						  double priorEstimation, double threshold, boolean resizingEnabled, 
        		 						  int windowResizePolicy, int estimatorID){
        	 		System.out.println("Creating estimator #"+estimatorID+" with Window size: "+initialSize);
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
        	 	
        	 	if(this.SizeWindow==-1) {
            	 	System.out.println("AGREGADOR VENTANA ESTATICA --> Adding result: "+value+"  window size was:"+this.SizeWindow);
                sum += value;
                len++;
        	 		
        	 	} else {
            	 	System.out.println("AGREGADOR VENTANA DINAMICA --> Adding result: "+value+"  window size was:"+this.SizeWindow);

	        	 	// Always storing extra results just in case we increment the window size
	  			this.window.add(value);
	  			// Remove oldest if it surpasses windowSize + increments
	         	if(!(window.size()<(this.SizeWindow+this.sizeIncrements))) // TODO. If increments are too big, the window will grow faster than the data arrives. should we do anything?
	         		this.window.remove(0);
	         	
	         	// Resize window
	         	if (this.resizingEnabled) updateWindowSize();
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
         
         // Returns the latest error window creating a sublist of the last errors.
         // As we store the plus last INCREMENT_SIZE errors, we need to start from the position INCREMENT_SIZE-1 (zero indexed)
         // Size of the sublist is WINDOW_SIZE 
         public double estimation(){ // = getLastWindowEstimation
        	 		System.out.println("In Estimator #"+this.estimatorID+"windowSize is "+SizeWindow+" so prediction is :"+(sum/len));
        	 		if (SizeWindow==-1) return sum/len;
        	 		else return estimateError(this.window.subList((this.window.size()-(this.sizeIncrements-1)), 
        	 				this.window.size() < this.SizeWindow ? this.window.size() : this.SizeWindow)); 
         }
                  
         public double getSmallerWindowEstimation (){
        	 		int smallerSize = this.SizeWindow-this.minimumSize;
     			return estimateError(this.window.subList((this.window.size()-(this.sizeIncrements-1)+this.minimumSize), 
     					 this.window.size() < smallerSize ? this.window.size() : smallerSize));
         }
         
         public double getLargeWindowEstimation (){
     			return estimateError(this.window);
         }
                 
         public double estimateError(List<Double> list) {
         	  Double sum = 0.0;
         	  if(!list.isEmpty()) {
         	    for (Double error : list) {
         	        sum += error;
         	    }
         	    return sum.doubleValue() / list.size();
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
	         				if(this.SizeWindow<this.minimumSize) 
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

    }

}


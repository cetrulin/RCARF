package moa.evaluation;
import java.util.ArrayList;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;

/**
 * Classification evaluator that updates evaluation results using a sliding
 * window.
 *
 * @author Andres L. Suarez-Cetrulo (suarezcetrulo at gmail dot com)
 * @version $Revision: 0 $
 */
public class DynamicWindowClassificationPerformanceEvaluator extends BasicClassificationPerformanceEvaluator {

    private static final long serialVersionUID = 1L;

    public IntOption widthOption = new IntOption("width",'w', "Size of Window", 10);
    public IntOption widthIncrementsOption = new IntOption("increments",'w', "Increments in Size of Window", 1);
    public IntOption minWidthOption = new IntOption("minWidth",'w', "Minimum Size of Window", 5);
    public FloatOption thresholdOption = new FloatOption("threshold",'w', "Threshold for resizing", 0.1);

    @Override
    protected Estimator newEstimator() {
    		System.out.println("NEW WINDOW ESTIMATOR CREATED");
        return new DynamicWindowEstimator(this.widthOption.getValue(),this.widthIncrementsOption.getValue(),this.minWidthOption.getValue(),this.thresholdOption.getValue());
    }

    public class DynamicWindowEstimator implements Estimator {

        /**
		 * by suarezcetrulo
		 */
		private static final long serialVersionUID = -6525799997539109003L;

		protected double[] window;

        protected int posWindow;

        protected int lenWindow;

        protected int SizeWindow; // AS windowSize
        
        // SUAREZCETRULO
        protected int minimumSize; 
        protected int defaultSize; 
        protected int sizeIncrements; 
        protected double threshold;
        
         ArrayList<Double> lastErrors;        
         
         // Constructor for window: threshold = -1 means that selected size update policy is 1
         public DynamicWindowEstimator(int initialSize,int sizeIncrements, int minSize, double threshold){
         		this.SizeWindow=initialSize;
         		this.defaultSize=initialSize;
         		this.sizeIncrements=sizeIncrements;
             	this.minimumSize=minSize;
         		this.threshold=threshold;
         		this.lastErrors=new ArrayList<Double>();
         }

         // It adds errors (1-accuracy) per classified row
         public void add(double iterarionError) {
         		// Always storing extra results just in case we increment the window size
         		if(lastErrors.size()<(this.SizeWindow+this.sizeIncrements)) {
         			this.lastErrors.add(iterarionError);
         		} else {
         			// Remove oldest
         			this.lastErrors.remove(0);
         		}
         }
         
         public void reset() {
         		this.lastErrors.clear();
         		this.SizeWindow=defaultSize;
         }
         
         public void clear() {
         		this.lastErrors.clear();
         }
         
         // Returns the latest error window creating a sublist of the last errors.
         // As we store the plus last INCREMENT_SIZE errors, we need to start from the position INCREMENT_SIZE-1 (zero indexed)
         // Size of the sublist is WINDOW_SIZE 
         public double estimation(){ // = getLastWindowEstimation
         		return estimateError((ArrayList<Double>) this.lastErrors.subList((this.lastErrors.size()-(this.sizeIncrements-1)), this.SizeWindow));
         }
                  
         public double getSmallerWindowEstimation (){
     			return estimateError((ArrayList<Double>) this.lastErrors.subList((this.lastErrors.size()-(this.sizeIncrements-1)+this.minimumSize), this.SizeWindow-this.minimumSize));
         }
         
         public double getLargeWindowEstimation (){
     			return estimateError(this.lastErrors);
         }
                 
         public double estimateError(ArrayList<Double> errors) {
         	  Double sum = 0.0;
         	  if(!errors.isEmpty()) {
         	    for (Double error : errors) {
         	        sum += error;
         	    }
         	    return sum.doubleValue() / errors.size();
         	  }
         	  return sum;         	  
         }
        
         public void windowResize(double priorError) {
      			// size update policy 2
        	 		if(priorError==-2) {
        	 			//TODO
        	 		}
         		// size update policy 2
         		if(priorError==-1) { // [ min(error(c,w_c)) - priorError ] > threshold -> window size increases
         			// OR if(getAverageOfErrors(getLastWindow()) - priorError > this.threshold) TODO to decide
     			if(estimation() - priorError > this.threshold)
     				this.SizeWindow+=this.sizeIncrements;
     			else // otherwise it decreases
     				this.SizeWindow-=this.sizeIncrements;
     				if(this.SizeWindow<this.minimumSize) 
     					this.SizeWindow=this.minimumSize;
         		} // size update policy 1
         		else { //For each c, W_c=s  , where s is an small number / For each iteration, independently of ny situations / a = active model
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
         		}      		
         }
        

    }

}


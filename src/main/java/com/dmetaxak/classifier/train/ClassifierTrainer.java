package com.dmetaxak.classifier.train;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.StringReader;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Properties;
import java.util.Set;












//import org.apache.commons.logging.Log;
//import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import edu.cmu.lemurproject.WarcRecord;
import edu.cmu.lemurproject.WritableWarcRecord;
import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.classify.LogisticClassifier;
import edu.stanford.nlp.classify.LogisticClassifierFactory;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.objectbank.ObjectBank;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.util.ArrayUtils;
import edu.stanford.nlp.util.HashIndex;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.StringUtils;

/*
 * 
 */
public class ClassifierTrainer {

//	static Index<String> featureIndex = new HashIndex<String>(); 
	static double[] WEIGHTS;
	static String[] classes = {"0","1"};
	static Index<String> FEATURES;
	public static LogisticClassifier<String, String> classifier; //initialized below
		
	public static void trainClassifier(String[] args) {
		
		//how to get current working directory
//		System.out.println(Paths.get(".").toAbsolutePath().normalize().toString());
		
		
		Properties prop = StringUtils.argsToProperties(args); // reads in regularization value and other stuff from input args (default value is 0.0)
	    double l1reg = Double.parseDouble(prop.getProperty("l1reg","0.0"));

	    Dataset<String, String> ds = new Dataset<String, String>(); //this will be all the data
	    
	 // read in all examples, one per line
	    System.out.println("Reading examples...");
	    for (String line : ObjectBank.getLineIterator(new File( "forjava.txt" ))) {
	    	line = line.replaceAll("[^a-zA-Z0-9 ]", "");
	    	String[] bits = line.split("\\s+"); //split on all whitespace (essentially tokenize?)
	    	Collection<String> features = new LinkedList<String>(Arrays.asList(bits).subList(1, bits.length));
	    	String label = bits[0]; //first character in the line is the label
	    	ds.add(features, label);
	    }
	    
//	    ds.summaryStatistics();

	    // TRAIN
	    boolean biased = prop.getProperty("biased", "false").equals("true"); //whether biased or not (default not)
	    LogisticClassifierFactory<String, String> factory = new LogisticClassifierFactory<String, String>();
	    classifier = factory.trainClassifier(ds, l1reg, 1e-4, biased);

		// CROSS VALIDATE
	    
	    // PRINT RELEVANT INFO 
	    
	    File file = new File("classification_results.txt");
	    FileWriter writer = null;
	    
	    WEIGHTS = classifier.getWeights();
	    int numweights = WEIGHTS.length;
	    FEATURES = classifier.getFeatureIndex();
	    int numfeatures = FEATURES.size();
	    
	    try {
	        writer = new FileWriter(file);
	        
	        // write the weights
	        writer.write( Integer.toString(numweights) + "\n");
		    for (int i=0; i < numweights; i++) {
		    	writer.write( Double.toString(WEIGHTS[i]) + "\n" );
		    }
	        
	        // write the features
//		    writer.write( Integer.toString(numfeatures) + "\n");
		    for (int i=0; i < numfeatures; i++) {
		    	writer.write( FEATURES.get(i) + "\n" );
		    }
		    
		    
	    } catch (IOException e) {
	        e.printStackTrace(); // I'd rather declare method with throws IOException and omit this catch.
	    } finally {
	        if (writer != null) try { writer.close(); } catch (IOException ignore) {}
	    }
	    System.out.printf("File is located at %s%n", file.getAbsolutePath());

	}
	
	public static void main(String[] args) {
		trainClassifier(args);
	}

	
}
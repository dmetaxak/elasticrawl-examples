package com.rossfairbanks.elasticrawl.examples;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Scanner;
import java.util.Set;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

import edu.cmu.lemurproject.WarcRecord;
import edu.cmu.lemurproject.WritableWarcRecord;
import edu.stanford.nlp.classify.LogisticClassifier;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.util.ArrayUtils;
import edu.stanford.nlp.util.HashIndex;
import edu.stanford.nlp.util.Index;

/*
 * A Hadoop custom mapper that implements the standard Word Count example.
 * 
 * Parses Common Crawl WET (WARC Encoded Text) files.
 */
public class WordCountMapper extends MapReduceBase 
	implements Mapper<LongWritable, WritableWarcRecord, Text, LongWritable> {

	private static final String COUNTER_GROUP = "Parser Mapper Counters";	
	private static final String RECORDS_FETCHED = "Records Fetched";
	private static final String RECORDS_COLLECTED = "Records Collected";

	private static final String WARC_RECORD_TYPE = "WARC-Type";
	private static final String WARC_URI = "WARC-Target-URI";
	private static final String CONVERSION_RECORD = "conversion";
	
	public static final Log LOG = LogFactory.getLog(WordCountMapper.class);

//	public static final String[] KEYWORDS = {"marijuana", "legalization", "cannabis"};
	
	public static LogisticClassifier<String, String> classifier; //initialized below
	
	// INITIALIZE CLASSIFIER
//	static {
//	    try {
//	    	Scanner weightsFile = new Scanner(new File("weights.txt"));	    	
//	    	weightsFile.useDelimiter(",");
//	    	
//	    	ArrayList<Double> temps = new ArrayList<Double>(); //accumulate weights in arraylist first
//	    	Index<String> featureIndex = new HashIndex<String>(); //accumulate features... 
//	    	//TODO: unclear to me how the weights and features are getting associated with each other?
//	    	
//	    	while (weightsFile.hasNext()) { //go through weights file
//	    		double w = Double.parseDouble(weightsFile.next()); //first item on each line is 
//	    		String f = weightsFile.next(); // second item is the feature
//	    		temps.add(w);
//	    		featureIndex.add(f);
//	    	}
//	    	weightsFile.close();
//	    	
//	    	Double[] d = new Double[temps.size()];
//	    	temps.toArray(d);
//	    	double[] WEIGHTS = ArrayUtils.toPrimitive(d); // make temp arraylist into an array (now that i know how big it is)
//	    	
//	    	String[] classes = {"0","1"};
//	    	
//	    	classifier = new LogisticClassifier<String, String>(WEIGHTS, featureIndex, classes);
//	    	
//	    } catch (FileNotFoundException e) {
//	    	LOG.info("!!!Couldn't import weights!!!");
//	    }
//	}
	
	
	/*
	 * Map method processes conversion records. Each WarcRecord contains the
	 * text content of a web page in the Common Crawl.
	 */
	public void map(LongWritable key, WritableWarcRecord value,
			OutputCollector<Text,LongWritable> output, Reporter reporter) throws IOException,FileNotFoundException {
		
		LOG.info("Initializing Classifier");
		//// make new classifier every time 
	    try {
	    	Scanner weightsFile = new Scanner(new File("weights.txt"));	    	
	    	weightsFile.useDelimiter(",");
	    	
	    	ArrayList<Double> temps = new ArrayList<Double>(); //accumulate weights in arraylist first
	    	Index<String> featureIndex = new HashIndex<String>(); //accumulate features... 
	    	//TODO: unclear to me how the weights and features are getting associated with each other?
	    	
	    	while (weightsFile.hasNext()) { //go through weights file
	    		double w = Double.parseDouble(weightsFile.next()); //first item on each line is 
	    		String f = weightsFile.next(); // second item is the feature
	    		temps.add(w);
	    		featureIndex.add(f);
	    	}
	    	weightsFile.close();
	    	
	    	Double[] d = new Double[temps.size()];
	    	temps.toArray(d);
	    	double[] WEIGHTS = ArrayUtils.toPrimitive(d); // make temp arraylist into an array (now that i know how big it is)
	    	
	    	String[] classes = {"0","1"};
	    	
	    	classifier = new LogisticClassifier<String, String>(WEIGHTS, featureIndex, classes);
	    	
	    } catch (FileNotFoundException e) {
	    	LOG.info("!!!Couldn't import weights!!!");
	    }
	    LOG.info("Classifier Initialized");
		
		////
		
		int recordCount = 0;
		
		// Get Warc record from the writable wrapper.
		WarcRecord record = value.getRecord();
		String recordType = record.getHeaderMetadataItem(WARC_RECORD_TYPE);
		String url = record.getHeaderMetadataItem(WARC_URI);
		
		if (recordType.equals(CONVERSION_RECORD)) { //TODO what does this mean? leftover from example
			// Get extracted page text.
			String pageText = record.getContentUTF8();
		
			// Remove all punctuation.
			pageText = pageText.replaceAll("[^a-zA-Z0-9 ]", "");
			
			// Normalize whitespace to single spaces.
			pageText = pageText.replaceAll("\\s+", " ");
			
			LOG.info("Tokenizing text...");
			// Tokenize the text into an arraylist of string features
			Reader reader=new StringReader(pageText);
			java.util.List<Word> words = PTBTokenizer.newPTBTokenizer(reader).tokenize();
			Set<String> newFeatures = new HashSet<String>();//use a set so duplicates aren't included
			for (Word w : words) {
				String toAdd = w.toString();
				//TODO: filter stopwords, strip punctuation, etc.
				newFeatures.add(toAdd);
			}
			
			LOG.info("Classifying text...");
			// Classify the text
			boolean toEmit = false;
			if (classifier.classOf(newFeatures) == "1") {
				toEmit = true; //if this website is in the YES class, emit it
			} //TODO: classifier only goes by unique set of features? no counts or anything?
						
			// Collect classified pages
			if (toEmit == true) {
				Text outputKey = new Text();
				LongWritable outputValue = new LongWritable();
				
				outputKey.set(url);
				outputValue.set(1);
				
				LOG.info("URL: " + url + ", onTopic: 1");
				
				output.collect(outputKey, outputValue);
				recordCount++; // not sure what this does? originally that was incremented for every word emitted but name suggests that it counts records
			} 
			else {
				Text outputKey = new Text();
				LongWritable outputValue = new LongWritable();
				
				outputKey.set(url);
				outputValue.set(0); //emit with a 0 value if not on topic
				
				LOG.info("URL: " + url + ", onTopic: 0");
				
				output.collect(outputKey, outputValue);
				recordCount++;
			}
			LOG.info("Done");
		}
			
		reporter.incrCounter(COUNTER_GROUP, RECORDS_FETCHED, 1);
		reporter.incrCounter(COUNTER_GROUP, RECORDS_COLLECTED, recordCount);
	}
}
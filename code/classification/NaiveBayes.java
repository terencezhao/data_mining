package classification;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created with IntelliJ IDEA.
 * User: Terence
 * Date: 11/21/13
 * Time: 10:10 PM
 * To change this template use File | Settings | File Templates.
 */
public class NaiveBayes {

    private static String myTrainFilePath = "code/a1a.train";
    private static String myTestFilePath = "code/a1a.test";
    private static String[] myArgs = new String[] {myTrainFilePath, myTestFilePath};

    public static void main(String [] args) {

    // Step 1 : Data I/O and Data Format (10 pts)
        ArrayList<ArrayList<String>> trainingData;
        ArrayList<ArrayList<String>> testData;

        ArrayList<ArrayList<String>> trainingInput;
        //ArrayList<ArrayList<String>> testInput;

        if(args.length == 2) {
            trainingData = inputData(args[0]);
            testData = inputData(args[1]);

            trainingInput = inputData(args[0]);
            //testInput = inputData(args[0]);

        } else {
            trainingData = inputData(myArgs[0]);
            testData = inputData(myArgs[1]);

            trainingInput = inputData(myArgs[0]);
            //testInput = inputData(myArgs[0]);
        }

    // Step 2 : Implement Basic Classification Method (20 pts for Option A, 10 pts for Option B)
        HashMap<String, ArrayList<HashMap<String, Integer>>> classifierFromTrainingData;
        classifierFromTrainingData = buildClassifierFromTrainingData(trainingInput);
        //HashMap<String, ArrayList<HashMap<String, Integer>>> classifierFromTestData;
        //classifierFromTestData = buildClassifierFromTrainingData(testInput);

        int[] trainingQuality = assignLabelsToTestDataUsingClassifier(trainingData, classifierFromTrainingData);
        int[] testQuality = assignLabelsToTestDataUsingClassifier(testData, classifierFromTrainingData);

    }

    // Takes in a file path and returns a 2D ArrayList of instances by attributes
    private static ArrayList<ArrayList<String>> inputData(String dataFilePath) {
        File dataFile = new File(dataFilePath);
        String dataFileAbsolutePath = dataFile.getAbsolutePath();
        ArrayList<ArrayList<String>> formatedData = new ArrayList<ArrayList<String>>();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(dataFile));
            String line;

            while ((line = reader.readLine()) != null) {
                ArrayList<String> instance = new ArrayList<String>();
                String[] parts = line.split(" ");
                for(String part : parts) {
                    instance.add(part);
                }
                formatedData.add(instance);
            }
            return formatedData;
        } catch(IOException ex) {
            throw new RuntimeException("Failed to read file", ex);
        }
    }

    public static HashMap<String, ArrayList<HashMap<String, Integer>>> buildClassifierFromTrainingData(ArrayList<ArrayList<String>> trainingData) {

        // Hashmap key is +1 and -1 where values are all instances
        // Instances are arraylists with integer
        // +1 : 0["count":4], 1["0":3 "1":6], 2["0":1, "1":3], 3[]...
        // -1 :
        HashMap<String, ArrayList<HashMap<String, Integer>>> classifier = new HashMap<String, ArrayList<HashMap<String, Integer>>>();


        for(ArrayList<String> instance : trainingData) {
            String label = instance.remove(0);

            // update label count
            if(!classifier.containsKey(label)) {
                classifier.put(label, new ArrayList<HashMap<String, Integer>>());
                classifier.get(label).add(0, new HashMap<String, Integer>());
                classifier.get(label).get(0).put("count", 1);
            } else {
                int count = classifier.get(label).get(0).get("count");
                classifier.get(label).get(0).put("count", count + 1);
            }

            int i = 1;
            for(String pair : instance) {
                // get pair index from left side of colon separator
                int index = Integer.valueOf(pair.substring(0, pair.indexOf(":")));

                // get pair value from right hand side of separator
                String value;
                while(i <= index) {
                    if(i == index) {
                        value = pair.substring(pair.indexOf(":") + 1, pair.length());
                    } else {
                        value = "0";
                    }

                    // update index value count
                    if(classifier.get(label).size() < i + 1) {
                        classifier.get(label).add(i, new HashMap<String, Integer>());
                    }
                    if(!classifier.get(label).get(i).containsKey(value)) {
                        classifier.get(label).get(i).put(value, 1);
                    } else {
                        int count = classifier.get(label).get(i).get(value);
                        classifier.get(label).get(i).put(value, count + 1);
                    }
                    i++;
                }
            }
        }

        // Testing
//        ArrayList<HashMap<String, Integer>> indexes = classifier.get("-1");
//        for(HashMap<String, Integer> attributeCount : indexes) {
//
//            System.out.print(indexes.indexOf(attributeCount) + ": ");
//
//            Iterator it = attributeCount.entrySet().iterator();
//            while (it.hasNext()) {
//                Map.Entry pairs = (Map.Entry)it.next();
//                System.out.print(pairs.getKey() + "=" + pairs.getValue() + " ");
//                it.remove(); // avoids a ConcurrentModificationException
//            }
//            System.out.println();
//        }
        return classifier;

    }

    public static int[] assignLabelsToTestDataUsingClassifier(ArrayList<ArrayList<String>> testData, HashMap<String, ArrayList<HashMap<String, Integer>>> classifier) {

        float positiveClassificationCount = (float)classifier.get("+1").get(0).get("count");
        float negativeClassificationCount = (float)classifier.get("-1").get(0).get("count");
        float totalCount = positiveClassificationCount + negativeClassificationCount;
        float probabilityOfPositiveClassification = positiveClassificationCount/totalCount;
        float probabilityOfNegativeClassification = negativeClassificationCount/totalCount;

        int truePositive = 0;
        int falsePositive = 0;
        int trueNegative = 0;
        int falseNegative = 0;


        for(ArrayList<String> instance : testData) {

            String label = instance.remove(0);

            float probabilityOfInstanceGivenPositive = 1;
            float probabilityOfInstanceGivenNegative = 1;

            int i = 1;
            for(String pair : instance) {
                int index = Integer.valueOf(pair.substring(0, pair.indexOf(":")));
                String value;
                while(i <= index) {
                    if(i == index) {
                        value = pair.substring(pair.indexOf(":") + 1, pair.length());
                    } else {
                        value = "0";
                    }

                    boolean indexInBounds = classifier.get("+1").size() > i && classifier.get("-1").size() > i;

                    if(indexInBounds) {
                        if(classifier.get("+1").get(i).containsKey(value) && classifier.get("-1").get(i).containsKey(value)) {
                            float positiveAttributeCount = (float)classifier.get("+1").get(i).get(value);
                            probabilityOfInstanceGivenPositive *= ( positiveAttributeCount / positiveClassificationCount );

                            float negativeAttributeCount = (float)classifier.get("-1").get(i).get(value);
                            probabilityOfInstanceGivenNegative *= ( negativeAttributeCount / negativeClassificationCount );
                        }
                    }

                    i++;
                }
            }
            float positiveClassification = probabilityOfInstanceGivenPositive * probabilityOfPositiveClassification;
            float negativeClassification = probabilityOfInstanceGivenNegative * probabilityOfNegativeClassification;
            float max = Math.max(positiveClassification, negativeClassification);
            if(max == positiveClassification) {
                //System.out.print("Classified as +1 : " + "Actually labeled " + label + " : ");
                if(label.equals("+1")) {
                    //System.out.println("True Positive");
                    truePositive += 1;
                } else if (label.equals("-1")) {
                    //System.out.println("False Positive");
                    falsePositive += 1;
                }
            } else if (max == negativeClassification) {
                //System.out.print("Classified as -1 : " + "Actually labeled " + label + " : ");
                if(label.equals("-1")) {
                    //System.out.println("True Negative");
                    trueNegative += 1;
                } else if (label.equals("+1")) {
                    //System.out.println("False Negative");
                    falseNegative += 1;
                }
            }
        }

        System.out.println(truePositive + " " + falseNegative + " " + falsePositive + " " + trueNegative);
        int[] quality = {truePositive, falseNegative, falsePositive, trueNegative};
        return quality;

    }
}

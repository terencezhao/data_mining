package classification;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by Terence on 12/4/13.
 */
public class NBAdaBoost {

    private static String myTrainFilePath = "a1a.train";
    private static String myTestFilePath = "a1a.test";
    private static String[] myArgs = new String[] {myTrainFilePath, myTestFilePath};
    private static int k = 5;

    public static void main(String [] args) {

        ArrayList<Tuple> trainingData;
        ArrayList<Tuple> testData;

        if(args.length == 2) {
            trainingData = formatData(args[0]);
            testData = formatData(args[1]);
        } else {
            trainingData = formatData(myArgs[0]);
            testData = formatData(myArgs[1]);
        }

        CompositeModel compositeModel = adaBoost(trainingData, k);

        ArrayList<String> trainingClassifications = compositeModel.classify(trainingData);
        ArrayList<String> testClassifications = compositeModel.classify(testData);

        qualify(trainingData, trainingClassifications);
        qualify(testData, testClassifications);
    }

    private static ArrayList<Tuple> formatData(String dataFilePath) {
        File dataFile = new File(dataFilePath);
        String dataFileAbsolutePath = dataFile.getAbsolutePath();
        ArrayList<Tuple> formattedData = new ArrayList<Tuple>();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(dataFile));
            String line;
            while ((line = reader.readLine()) != null) {
                Tuple tuple = new Tuple(line);
                formattedData.add(tuple);
            }
            return formattedData;
        } catch(IOException ex) {
            throw new RuntimeException("Failed to read file", ex);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //  ADABOOST ALGORITHM (source : DataMiningConcepts and Techniques 2nd Ed. Page 368-370)
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    private static CompositeModel adaBoost(ArrayList<Tuple> D, int k) {
        CompositeModel compositeModel = new CompositeModel();
        initializeWeights(D);
        ArrayList<Tuple> Di = D;
        for(int i = 0; i < k; i++) {
            Di = sampleWithReplacement(Di);
            Model Mi = new Model(Di);
            Double error = computeError(Mi, Di);
            Mi.setErrorRate(error);
            if(error > 0.5) {
                initializeWeights(Di);
                i--;
                continue;
            }
            for(Tuple tuple : Di) {
                String classification = Mi.classify(tuple);
                String label = tuple.getLabel();
                if(classification.equals(label)) {
                    double updateWeight = tuple.getWeight() * ( error / (1.0 - error) );
                    tuple.setWeight(updateWeight);
                }
            }
            normalizeWeights(Di);
            compositeModel.addModel(Mi);
        }
        return compositeModel;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //  ADABOOST ALGORITHM HELPER FUNCTIONS
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    private static void initializeWeights(ArrayList<Tuple> D) {
        for(Tuple tuple : D) {
            tuple.setWeight(1.0/D.size());
        }
    }

    private static void normalizeWeights(ArrayList<Tuple> Di) {
        double sumWeight = 0.0;
        for(Tuple tuple : Di) {
            sumWeight = sumWeight + tuple.getWeight();
        }
        for(Tuple tuple : Di) {
            double normalizedWeight = (tuple.getWeight() / sumWeight) * 1.0;
            tuple.setWeight(normalizedWeight);
        }
    }

    // Referenced http://stackoverflow.com/questions/6737283/weighted-randomness-in-java
    private static ArrayList<Tuple> sampleWithReplacement(ArrayList<Tuple> D) {
        double totalWeight = 0.0;
        for(Tuple tuple : D) {
            totalWeight += tuple.getWeight();
        }
        ArrayList<Tuple> Di = new ArrayList<Tuple>();
        while(Di.size() < D.size()) {
            int randomIndex = -1;
            double random = Math.random() * totalWeight;
            for(Tuple tuple : D) {
                random -= tuple.getWeight();
                if(random <= 0.0d) {
                    randomIndex = D.indexOf(tuple);
                    break;
                }
            }
            if(randomIndex != -1) {
                Di.add(new Tuple(D.get(randomIndex)));
            }
        }
        return Di;
    }

    private static double computeError(Model model, ArrayList<Tuple> Di) {
        double error = 0.0;
        for(Tuple tuple : Di) {
            String label = tuple.getLabel();
            String classification = model.classify(tuple);
            if(classification.equals(label)) {
                error = error + (tuple.getWeight() * 0.0);
            } else {
                error = error + (tuple.getWeight() * 1.0);
            }
        }
        return error;
    }

    private static void qualify(ArrayList<Tuple> dataSet, ArrayList<String> classifications) {
        int truePositive = 0;
        int falseNegative = 0;
        int falsePositive = 0;
        int trueNegative = 0;
        for(int i = 0; i < dataSet.size(); i++) {
            String givenLabel = dataSet.get(i).getLabel();
            String predictedLabel = classifications.get(i);
            if(predictedLabel.equals("+1") && givenLabel.equals("+1")) {
                truePositive += 1;
            }
            if(predictedLabel.equals("-1") && givenLabel.equals("+1")) {
                falseNegative += 1;
            }
            if(predictedLabel.equals("+1") && givenLabel.equals("-1")) {
                falsePositive += 1;
            }
            if(predictedLabel.equals("-1") && givenLabel.equals("-1")) {
                trueNegative += 1;
            }
        }
        System.out.println(truePositive + " " + falseNegative + " " + falsePositive + " " + trueNegative);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // NAIVE BAYES CLASSIFIER
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    public static class Model {

        double errorRate;

        int positiveCount;
        int negativeCount;
        HashMap<String, Integer> positiveRecords;
        HashMap<String, Integer> negativeRecords;

        public Model(ArrayList<Tuple> trainingSet) {

            positiveCount = 0;
            negativeCount = 0;
            positiveRecords = new HashMap<String, Integer>();
            negativeRecords = new HashMap<String, Integer>();

            for(Tuple tuple : trainingSet) {
                if(tuple.getLabel().equals("+1")) {
                    positiveCount += 1;
                    ArrayList<Pair> pairs = tuple.getPairs();
                    for(Pair pair : pairs) {
                        String key = pair.toString();
                        if(!positiveRecords.containsKey(key)) {
                            positiveRecords.put(key, 1);
                        } else {
                            int count = positiveRecords.get(key);
                            positiveRecords.put(key, count + 1);
                        }
                    }
                }
                else if(tuple.getLabel().equals("-1")) {
                    negativeCount += 1;
                    ArrayList<Pair> pairs = tuple.getPairs();
                    for(Pair pair : pairs) {
                        String key = pair.toString();
                        if(!negativeRecords.containsKey(key)) {
                            negativeRecords.put(key, 1);
                        } else {
                            int count = negativeRecords.get(key);
                            negativeRecords.put(key, count + 1);
                        }
                    }
                }
            }
        }

        public String classify(Tuple tuple) {
            double totalCount = positiveCount + negativeCount;
            double probabilityOfPositiveClassification = positiveCount / totalCount;
            double probabilityOfNegativeClassification = negativeCount / totalCount;
            double probabilityOfTupleGivenPositive = 1.0;
            double probabilityOfTupleGivenNegative = 1.0;
            ArrayList<Pair> pairs = tuple.getPairs();
            for(Pair pair : pairs) {
                boolean indexInBounds = pairs.indexOf(pair) < positiveRecords.size() && pairs.indexOf(pair) < negativeRecords.size();
                if(indexInBounds) {
                    String key = pair.toString();
//                    if(positiveRecords.containsKey(key) && negativeRecords.containsKey(key) ) {
//                        double positiveAttributeCount = positiveRecords.get(key);
//                        probabilityOfTupleGivenPositive *= positiveAttributeCount / positiveCount;
//                        double negativeAttributeCount = negativeRecords.get(key);
//                        probabilityOfTupleGivenNegative *= negativeAttributeCount / negativeCount;
//                    }
                    if(positiveRecords.containsKey(key)) {
                        double positiveAttributeCount = positiveRecords.get(key);
                        probabilityOfTupleGivenPositive *= positiveAttributeCount / positiveCount;
                    }
                    else {
                        probabilityOfTupleGivenPositive *= 1 / positiveCount + 1;
                    }
                    if(negativeRecords.containsKey(key)) {
                        double negativeAttributeCount = negativeRecords.get(key);
                        probabilityOfTupleGivenNegative *= negativeAttributeCount / negativeCount;
                    }
                    else {
                        probabilityOfTupleGivenNegative *= 1 / negativeCount + 1;
                    }
                }
            }
            double positiveClassification = probabilityOfTupleGivenPositive * probabilityOfPositiveClassification;
            double negativeClassification = probabilityOfTupleGivenNegative * probabilityOfNegativeClassification;
            double max = Math.max(positiveClassification, negativeClassification);
            String label;
            if(max == positiveClassification) {
                label = "+1";
            } else {
                label =  "-1";
            }
            return label;
        }

        public void setErrorRate(double error) {
            this.errorRate = error;
        }
        public double getErrorRate() {
            return errorRate;
        }

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // COMPOSITE MODEL
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    public static class CompositeModel {
        ArrayList<Model> models;
        public CompositeModel() {
            models = new ArrayList<Model>();
        }

        public void addModel(Model model) {
            models.add(model);
        }

        public ArrayList<String> classify(ArrayList<Tuple> testData) {
            ArrayList<String> classifications = new ArrayList<String>();
            for(Tuple tuple : testData) {
                Double positiveWeight = 0.0;
                Double negativeWeight = 0.0;
                for(Model model : models) {
                    double classifierWeight = Math.log((1.0 - model.getErrorRate())/model.getErrorRate());
                    String classPrediction = model.classify(tuple);
                    if(classPrediction.equals("+1")) {
                        positiveWeight += classifierWeight;
                    } else if(classPrediction.equals("-1")) {
                        negativeWeight += classifierWeight;
                    }
                }
                //Tuple resultTuple = new Tuple(tuple);
                if(positiveWeight > negativeWeight) {
                    classifications.add("+1");
                } else {
                    classifications.add("-1");
                }
            }
            return classifications;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // HELPER CLASSES
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    public static class Tuple {

        private String label;
        private ArrayList<Pair> pairs;
        private double weight;

        public Tuple(Tuple tuple) {
            this.label = tuple.label;
            this.weight = tuple.weight;
            this.pairs = new ArrayList<Pair>();
            for(Pair pair : tuple.pairs) {
                pairs.add(new Pair(pair));
            }
        }

        public Tuple(String instance) {
            String[] parts = instance.split(" ");
            this.label = parts[0];
            pairs = new ArrayList<Pair>();
            int j = 1;
            for(int i = 1; i < parts.length; i++) {
                String[] part = parts[i].split(":");
                int index = Integer.valueOf(part[0]);
                int value = Integer.valueOf(part[1]);
                // fill missing pairs with value 0
                while(j < index) {
                    Pair pair = new Pair(j, 0);
                    pairs.add(pair);

                    j++;
                }
                Pair pair = new Pair(index, value);
                pairs.add(pair);
                j++;
            }
        }

        public void setWeight(double weight) {
            this.weight = weight;
        }
        public double getWeight() {
            return weight;
        }

        public String getLabel() {
            return label;
        }
        public void setLabel(String label) {
            this.label = label;
        }
        public ArrayList<Pair> getPairs() {
            return pairs;
        }
    }

    public static class Pair {
        private int index;
        private int value;

        public Pair(int index, int value) {
            this.index = index;
            this.value = value;
        }
        public Pair(Pair pair) {
            this.index = pair.index;
            this.value = pair.value;
        }
        public void setIndex(int index) {
            this.index = index;
        }
        public void setValue(int value) {
            this.value = value;
        }
        public int getIndex() {
            return index;
        }
        public int getValue() {
            return value;
        }
        public String toString() {
            return String.valueOf(index)+ ":" + String.valueOf(value);
        }
    }
}

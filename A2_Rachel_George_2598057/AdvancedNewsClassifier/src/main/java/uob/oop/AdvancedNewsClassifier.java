package uob.oop;

import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class AdvancedNewsClassifier {
    public Toolkit myTK = null;
    public static List<NewsArticles> listNews = null;
    public static List<Glove> listGlove = null;
    public List<ArticlesEmbedding> listEmbedding = null;
    public MultiLayerNetwork myNeuralNetwork = null;

    public final int BATCHSIZE = 10;

    public int embeddingSize = 0;
    private static StopWatch mySW = new StopWatch();

    public void wordVector(String vocab, double vector) {
    }


    public AdvancedNewsClassifier() throws IOException {
        myTK = new Toolkit();
        myTK.loadGlove();
        listNews = myTK.loadNews();
        listGlove = createGloveList();
        listEmbedding = loadData();
    }

    public static void main(String[] args) throws Exception {
        mySW.start();
        AdvancedNewsClassifier myANC = new AdvancedNewsClassifier();

        myANC.embeddingSize = myANC.calculateEmbeddingSize(myANC.listEmbedding);
        myANC.populateEmbedding();
        myANC.myNeuralNetwork = myANC.buildNeuralNetwork(2);
        myANC.predictResult(myANC.listEmbedding);
        myANC.printResults();
        mySW.stop();
        System.out.println("Total elapsed time: " + mySW.getTime());
    }


    public List<Glove> createGloveList() {
        List<Glove> listResult = new ArrayList<>();
        //TODO Task 6.1 - 5 Marks
        List<String> stopwords = List.of(Toolkit.STOPWORDS);
        for (int i = 0; i < Toolkit.listVocabulary.size(); i++) {
            String vocab = Toolkit.listVocabulary.get(i);
            Vector vector = new Vector(Toolkit.listVectors.get(i));
            if (!stopwords.contains(vocab.toLowerCase())) {
                Glove glove = new Glove(vocab, vector);
                listResult.add(glove);
            }
        }
        return listResult;
    }

    public static List<ArticlesEmbedding> loadData() {
        List<ArticlesEmbedding> listEmbedding = new ArrayList<>();
        for (NewsArticles news : listNews) {
            ArticlesEmbedding myAE = new ArticlesEmbedding(news.getNewsTitle(), news.getNewsContent(), news.getNewsType(), news.getNewsLabel());
            listEmbedding.add(myAE);
        }
        return listEmbedding;
    }

    public int calculateEmbeddingSize(List<ArticlesEmbedding> _listEmbedding) {
        int intMedian = -1;
        //TODO Task 6.2 - 5 Marks

        List<Integer> lengthList = new ArrayList<>();

        for (int i = 0; i < _listEmbedding.size(); i++) {
            int counter = 0;
            String[] content = _listEmbedding.get(i).getNewsContent().split(" ");
            for (String eachWord : content) {
                if (Toolkit.getListVocabulary().contains(eachWord)) {
                    counter += 1;
                }
            }
            lengthList.add(counter);
        }
        lengthList.sort(null);

        if ((lengthList.size() % 2) == 0) {
            intMedian = (lengthList.get((lengthList.size() / 2)) + lengthList.get((((lengthList.size() / 2) + 1)))) / 2;
        } else {
            intMedian = lengthList.get((lengthList.size() + 1) / 2);
        }

        return intMedian;
    }

    public void populateEmbedding() {
        //TODO Task 6.3 - 10 Marks
        for (ArticlesEmbedding articlesEmbedding : listEmbedding) {
            try {
                INDArray eachArticleEmbedding = articlesEmbedding.getEmbedding();
            } catch (InvalidSizeException e) {
                articlesEmbedding.setEmbeddingSize(embeddingSize);
            } catch (InvalidTextException e) {
                articlesEmbedding.getNewsContent();
            } catch (Exception e) {
                e.getMessage();
            }
        }
    }

    public DataSetIterator populateRecordReaders(int _numberOfClasses) throws Exception {
        ListDataSetIterator myDataIterator = null;
        List<DataSet> listDS = new ArrayList<>();
        INDArray inputNDArray = null;
        INDArray outputNDArray = null;

        //TODO Task 6.4 - 8 Marks

        for (ArticlesEmbedding articlesEmbedding : listEmbedding) {
            if ("Training".equalsIgnoreCase(articlesEmbedding.getNewsType().toString())) {
                inputNDArray = articlesEmbedding.getEmbedding();
                outputNDArray = Nd4j.zeros(1, _numberOfClasses);
                int index = Integer.parseInt(articlesEmbedding.getNewsLabel());
                if (index != -1) {
                    outputNDArray.putScalar(index - 1, 1);
                }
                DataSet myDataSet = new DataSet(inputNDArray, outputNDArray);
                listDS.add(myDataSet);
            }
        }
        return new ListDataSetIterator(listDS, BATCHSIZE);
    }

    public MultiLayerNetwork buildNeuralNetwork(int _numOfClasses) throws Exception {
        DataSetIterator trainIter = populateRecordReaders(_numOfClasses);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(Adam.builder().learningRate(0.02).beta1(0.9).beta2(0.999).build())
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(embeddingSize).nOut(15)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.HINGE)
                        .activation(Activation.SOFTMAX)
                        .nIn(15).nOut(_numOfClasses).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        for (int n = 0; n < 100; n++) {
            model.fit(trainIter);
            trainIter.reset();
        }
        return model;
    }

    public List<Integer> predictResult(List<ArticlesEmbedding> _listEmbedding) throws Exception {
        List<Integer> listResult = new ArrayList<>();
        //TODO Task 6.5 - 8 Marks
        for (ArticlesEmbedding article : _listEmbedding) {
            if ("Testing".equalsIgnoreCase(article.getNewsType().toString())) {
                int[] labelGroupArray = myNeuralNetwork.predict(article.getEmbedding());
                for (int i = 0; i < labelGroupArray.length; i++) {
                    listResult.add(labelGroupArray[i]);
                    article.setNewsLabel(String.valueOf(labelGroupArray[i]));
                }
            }
        }
        return listResult;
    }

    public void printResults() {
        //TODO Task 6.6 - 6.5 Marks
        List<String> newsLables = new ArrayList<>();
        int newsGroups = 1;
        for (ArticlesEmbedding article : listEmbedding) {
            if ("Testing".equalsIgnoreCase(article.getNewsType().toString())) {
                newsLables.add(article.getNewsLabel());
            }
        }

        newsLables.sort(null);
        for (int i = 1; i < newsLables.size(); i++) {
            if(!(newsLables.get(i).equalsIgnoreCase(newsLables.get(i-1)))){
                newsGroups++;
            }
        }

        for (int i = 0; i < newsGroups; i++) {
            System.out.println("Group "+(i+1));
            for (ArticlesEmbedding article : listEmbedding) {
                if ("Testing".equalsIgnoreCase(article.getNewsType().toString())) {
                    if (article.getNewsLabel().equalsIgnoreCase(String.valueOf(i))){
                        System.out.println(article.getNewsTitle());
                    }
                }
            }
        }
    }
}


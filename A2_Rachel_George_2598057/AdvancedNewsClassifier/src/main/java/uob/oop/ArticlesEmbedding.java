package uob.oop;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.Properties;


public class ArticlesEmbedding extends NewsArticles {
    private int intSize = -1;
    private String processedText = "";

    private INDArray newsEmbedding = Nd4j.create(0);

    public ArticlesEmbedding(String _title, String _content, NewsArticles.DataType _type, String _label) {
        //TODO Task 5.1 - 1 Mark
        super(_title,_content,_type,_label);
    }

    public void setEmbeddingSize(int _size) {
        //TODO Task 5.2 - 0.5 Marks
        this.intSize = _size;
    }

    public int getEmbeddingSize(){
        return intSize;
    }
    private boolean worked_once = false;

    @Override
    public String getNewsContent() {
        if (processedText.isEmpty()) {
            //TODO Task 5.3 - 10 Marks

            String cleanedContent = textCleaning(super.getNewsContent());
            Properties properties = new Properties();
            properties.setProperty("annotators", "tokenize,pos,lemma");
            StanfordCoreNLP pipeline = new StanfordCoreNLP(properties);
            CoreDocument document = new CoreDocument(cleanedContent);
            pipeline.annotate(document);

            StringBuilder cleaningContent = new StringBuilder();
            for (CoreLabel token : document.tokens()) {
                cleaningContent.append(token.lemma()).append(" ");
            }
            String[] strDocument = cleaningContent.toString().split("\s+");

            StringBuilder removedStopwords = new StringBuilder();
            for (String word : strDocument) {
                boolean stopwordFound = false;
                for (String stopWord : Toolkit.STOPWORDS) {
                    if (word.equalsIgnoreCase(stopWord)) {
                        stopwordFound = true;
                        break;
                    }
                }
                if (!stopwordFound) {
                    removedStopwords.append(word).append(" ");
                }
            }
            processedText = removedStopwords.toString().toLowerCase();
        }
        return processedText.trim();
    }


    public INDArray getEmbedding() throws Exception {
        //TODO Task 5.4 - 20 Marks
        if (intSize == -1) {
            throw new InvalidSizeException("Invalid size");
        } else if (processedText.isEmpty()) {
            throw new InvalidTextException("Invalid text");
        } else if (newsEmbedding.isEmpty()) {
            String[] processedWordsArray = processedText.split(" ");
            newsEmbedding = Nd4j.create(intSize, AdvancedNewsClassifier.listGlove.get(0).getVector().getVectorSize());

            int nonzeroRows = 0;

            for (int i = 0; i < processedWordsArray.length && nonzeroRows < intSize; i++) {
                boolean gloveWord = false;
                int gloveIndex = -1;
                for (int j = 0; j < AdvancedNewsClassifier.listGlove.size(); j++) {
                    if (processedWordsArray[i].equalsIgnoreCase(AdvancedNewsClassifier.listGlove.get(j).getVocabulary())) {
                        gloveWord = true;
                        gloveIndex = j;
                        break;
                    }
                }
                if (gloveWord) {
                    double[] gloveVector = AdvancedNewsClassifier.listGlove.get(gloveIndex).getVector().getAllElements();
                    newsEmbedding.putRow(nonzeroRows++, Nd4j.create(gloveVector));
                }
            }
        }
        return Nd4j.vstack(newsEmbedding.mean(1));
    }


    /***
     * Clean the given (_content) text by removing all the characters that are not 'a'-'z', '0'-'9' and white space.
     * @param _content Text that need to be cleaned.
     * @return The cleaned text.
     */
    private static String textCleaning(String _content) {
        StringBuilder sbContent = new StringBuilder();

        for (char c : _content.toLowerCase().toCharArray()) {
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || Character.isWhitespace(c)) {
                sbContent.append(c);
            }
        }

        return sbContent.toString().trim();
    }
}

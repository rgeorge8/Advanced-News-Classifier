package uob.oop;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

public class Toolkit {
    public static List<String> listVocabulary = null;
    public static List<double[]> listVectors = null;
    private static final String FILENAME_GLOVE = "glove.6B.50d_Reduced.csv";

    public static final String[] STOPWORDS = {"a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", "and", "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", "cannot", "could", "dear", "did", "do", "does", "either", "else", "ever", "every", "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its", "just", "least", "let", "like", "likely", "may", "me", "might", "most", "must", "my", "neither", "no", "nor", "not", "of", "off", "often", "on", "only", "or", "other", "our", "own", "rather", "said", "say", "says", "she", "should", "since", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us", "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "yet", "you", "your"};

    public void loadGlove() throws IOException {
        BufferedReader myReader = null;
        //TODO Task 4.1 - 5 marks
        listVocabulary = new ArrayList<>();
        listVectors = new ArrayList<>();
        try {
            myReader = new BufferedReader(new FileReader(Toolkit.getFileFromResource(FILENAME_GLOVE)));
            String currentLine;
            while ((currentLine = myReader.readLine()) != null) {
                String[] fileArray = currentLine.split(",");
                listVocabulary.add(fileArray[0]);
                double[] vectorArray = new double[fileArray.length-1];
                for (int i = 0; i < fileArray.length-1; i++) {
                    vectorArray[i] = Double.parseDouble(fileArray[i+1]);
                }
                listVectors.add(vectorArray);
            }
            myReader.close();

        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    private static File getFileFromResource(String fileName) throws URISyntaxException {
        ClassLoader classLoader = Toolkit.class.getClassLoader();
        URL resource = classLoader.getResource(fileName);
        if (resource == null) {
            throw new IllegalArgumentException(fileName);
        } else {
            return new File(resource.toURI());
        }
    }

    public List<NewsArticles> loadNews() {
        List<NewsArticles> listNews = new ArrayList<>();
        //TODO Task 4.2 - 5 Marks

        Path newsPath = Paths.get("src\\main\\resources\\News");
        File fileName = newsPath.toFile();
        File[] files = fileName.listFiles();
        for (File file : files) {
            try {
                List<String> allLinesOfFile = Files.readAllLines(file.toPath());
                StringBuilder htms = new StringBuilder(String.join("\n", allLinesOfFile));
                if (file.getName().endsWith(".htm")) {
                    String htmFile = htms.toString();
                    String title = HtmlParser.getNewsTitle(htmFile);
                    String content = HtmlParser.getNewsContent(htmFile);
                    NewsArticles.DataType type = HtmlParser.getDataType(htmFile);
                    String label = HtmlParser.getLabel(htmFile);
                    NewsArticles article = new NewsArticles(title, content, type, label);
                    listNews.add(article);
                }
            } catch (Exception e) {
                System.out.println(e.getMessage());
            }
        }
        return listNews;
    }

    public static List<String> getListVocabulary() {
        return listVocabulary;
    }

    public static List<double[]> getlistVectors() {
        return listVectors;
    }
}

package play;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Iterables;
import com.medallia.word2vec.Word2VecModel;
import com.medallia.word2vec.Word2VecTrainerBuilder;
import com.medallia.word2vec.neuralnetwork.NeuralNetworkType;
import com.medallia.word2vec.util.Common;
import io.juptr.fusion.TextProcessing;
import io.juptr.fusion.stemming.Language;
import io.juptr.fusion.textprocessing.DocumentSet;
import io.juptr.fusion.textprocessing.DocumentSource;
import io.juptr.fusion.textprocessing.wiki.WikiArtifactBuilder;
import io.juptr.gravity.systemconfig.JuptrCfg;
import io.juptr.helper.JuptrUtil;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by ruedi on 30.01.16.
 */
public class TestWiki {

    public static void main(String[] args) throws InterruptedException, IOException {
        TextProcessing.init(JuptrCfg.read());

        Iterator<DocumentSource> documentSet = WikiArtifactBuilder.get()
            .cachedWikiDocuments(Language.EN)
            .streamDocs()
//            .limit(100)
            .iterator();

        Iterable<List<String>> corpus = new Iterable<List<String>>() {
            @Override
            public Iterator<List<String>> iterator() {
                return new Iterator<List<String>>() {
                    @Override
                    public boolean hasNext() {
                        return documentSet.hasNext();
                    }

                    int count;
                    @Override
                    public List<String> next() {
                        if (count++%1000==0)
                            System.out.println(count);
                        List<String> res = documentSet.next().idfLargerThan(4).collect(Collectors.toList());
                        return res;
                    }
                };
            }
        };

        Word2VecModel model =
            Word2VecModel.trainer()
                .setInitialLearningRate(0.05)
                .setMinVocabFrequency(30)
                .useNumThreads(8)
                .setWindowSize(10)
                .type(NeuralNetworkType.CBOW)
                .useHierarchicalSoftmax()
                .setLayerSize(150)
                .useNegativeSamples(5)
                .setDownSamplingRate(1e-3)
                .setNumIterations(15)
                .train(corpus);

        JuptrUtil.writeObject("w2vjava_en.oos",model.toThrift());
    }
}

package com.example;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class App {
    public static void main( String[] args ) throws IOException {
        INDArray features = Nd4j.read(new BufferedInputStream(Files.newInputStream(Paths.get("example-features.bin"))));
        INDArray labels = Nd4j.read(new BufferedInputStream(Files.newInputStream(Paths.get("example-labels.bin"))));
        DataSet dataSet = new DataSet(features, labels);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(0)
                .iterations(1)
                .miniBatch(true)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.RMSPROP)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .learningRate(1e-1)
                .regularization(true)
                .l2(3)
                .list(2).layer(0,new DenseLayer.Builder()
                        .nIn(features.shape()[1])
                        .nOut(1000)
                        .activation("hardtanh")
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(1000)
                        .nOut(labels.shape()[1])
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1));

        SplitTestAndTrain splitTestAndTrain = dataSet.splitTestAndTrain(0.65);

        for (DataSet batch : splitTestAndTrain.getTrain().batchBy(500)) {
            model.fit(batch);

        }

        Evaluation eval = new Evaluation(labels.shape()[1]);
        for (DataSet batch : splitTestAndTrain.getTest().batchBy(500)) {
            INDArray output = model.output(batch.getFeatureMatrix(), false);
            eval.eval(batch.getLabels(), output);
        }

        System.out.println(eval.stats());
    }
}

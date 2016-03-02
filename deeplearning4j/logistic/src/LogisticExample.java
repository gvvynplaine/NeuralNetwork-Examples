
import java.io.IOException;
import java.util.List;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.LoggerFactory;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;

public class LogisticExample {


	public static void main(String[] args) throws IOException, ClassNotFoundException {

		// disable logging
		Logger root = (Logger) LoggerFactory.getLogger(ch.qos.logback.classic.Logger.ROOT_LOGGER_NAME);
	    root.setLevel(Level.ERROR);
		
	    /* build the graph. Only one layer is created because this is 
	     * the lowest dl4j can go. 2 inputs are specified and 1 output.
	     * The sigmoid activation will perform the dot product of the 
	     * weights and bias.
	     */
		MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.list(1)
				.layer(0, new OutputLayer.Builder()
						.nIn(2)
						.nOut(1)
						.activation("sigmoid")
						.build())
				.backprop(true);
		MultiLayerNetwork model = new MultiLayerNetwork(builder.build());
		model.init();

		// specify the x0 and x2 values
		INDArray x = Nd4j.create(new double[][] { { -1.0, -2.0}, });
		
		// specify the weights. w0 and w1 are contained in W
		model.getLayers()[0].setParam("W", Nd4j.create(new double[][] {{2},{-3}}));
		model.getLayers()[0].setParam("b", Nd4j.create(new double[] {-3}));

		// process input data
		List<INDArray> results = model.feedForward(x.getRow(0));
		
		// get weights of the sigmoid
		INDArray weights = model.getLayers()[0].getParam("W");
		INDArray bias = model.getLayers()[0].getParam("b");
		
		System.out.println("x0, x1: " + x);
		System.out.println("w0, w1: " + weights);
		System.out.println("w3 (bias): " + bias);
		
		System.out.println("Result: " + results.get(1));
		
	}
}

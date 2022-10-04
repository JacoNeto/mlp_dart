import 'package:mlpdart/mlp/dataset.dart';
import 'package:mlpdart/mlp/layer.dart';
import 'package:mlpdart/mlp/mlp.dart';
import 'package:mlpdart/mlp/neuron.dart';

void main() {
  // Set Min and Max weight value for all neurons
  Neuron.setRange(-1, 1);

  // Create a Neural Network with 3 Layers
  MultiLayerPerception mlp =
      MultiLayerPerception(3); // 1 input + 1 hidden + 1 output
  // No need to add input layer, it will be added from dataset automatically
  mlp.layers[1] = Layer.hidden(
      6, 2); // Hidden layer / 6 neurons each have 2 weights (comlpections)
  mlp.layers[2] = Layer.hidden(
      1, 6); // Output layer / 1 neuron with 6 weights (comlpections)

  // Create the training data
  Dataset dataset = loadDataset(); // Hard-coded for now.

  print("============");
  print("Output before training");
  print("============");
  for (var i in dataset.pairs) {
    MultiLayerPerception.forward(mlp, i.inputData, bias: -0.1);
    print(
        'inputs: ${mlp.layers[0]!.neurons[0].value}, ${mlp.layers[0]!.neurons[1].value}');
    print('output: ${mlp.layers[2]!.neurons[0].value}');
  }

  MultiLayerPerception.train(mlp, dataset, 100000, 0.05, bias: -0.1);

  print("============");
  print("Output after training");
  print("============");
  for (var i in dataset.pairs) {
    MultiLayerPerception.forward(mlp, i.inputData, bias: -0.1);
    print(
        'inputs: ${mlp.layers[0]!.neurons[0].value}, ${mlp.layers[0]!.neurons[1].value}');
    print('output: ${mlp.layers[2]!.neurons[0].value}');
  }
}

// XOR Example
Dataset loadDataset() {
  // TODO: Make this more generic. / create a get function from yaml etc.
  List<double> input1 = [0, 0];
  List<double> input2 = [0, 1];
  List<double> input3 = [1, 0];
  List<double> input4 = [1, 1];

  List<double> expectedOutput1 = [0];
  List<double> expectedOutput2 = [1];
  List<double> expectedOutput3 = [1];
  List<double> expectedOutput4 = [0];

  var dataset = Dataset(); // Empty Dataset

  // Fill dataset with pairs of input and output data
  dataset.pairs.add(Pair(input1, expectedOutput1));
  dataset.pairs.add(Pair(input2, expectedOutput2));
  dataset.pairs.add(Pair(input3, expectedOutput3));
  dataset.pairs.add(Pair(input4, expectedOutput4));

  return dataset;
}

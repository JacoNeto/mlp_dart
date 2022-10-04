import 'dart:convert';
import 'dart:io';

import 'package:csv/csv.dart';
import 'package:mlpdart/mlp/dataset.dart';
import 'package:mlpdart/mlp/layer.dart';
import 'package:mlpdart/mlp/mlp.dart';
import 'package:mlpdart/mlp/neuron.dart';
import 'package:mlpdart/csv/csv_utils.dart';

void main() async {
  // Set Min and Max weight value for all neurons
  Neuron.setRange(-1, 1);

  // Create a Neural Network with 3 Layers
  MultiLayerPerceptron mlp =
      MultiLayerPerceptron(3); // 1 input + 1 hidden + 1 output
  // No need to add input layer, it will be added from dataset automatically
  mlp.layers[1] = Layer.hidden(
      8, 20); // Hidden layer / 8 neurons each have 20 weights (comlpections)
  mlp.layers[2] = Layer.hidden(
      3, 8); // Output layer / 3 neuron with 8 weights (comlpections)

  // Create the training dataset
  Dataset dataset = await loadDataset();
  // Create the testing dataset
  // Dataset testingDataset = await loadTestDataset();

  print("============");
  print("Output before training");
  print("============");
  print(dataset.pairs);
  for (Pair i in dataset.pairs) {
    MultiLayerPerceptron.forward(mlp, i.inputData, bias: 0);
    print("Inputs:");
    for (Neuron n in mlp.layers[0]!.neurons) {
      print("[${n.value}]");
    }
    print("Outputs:");
    for (Neuron n in mlp.layers[2]!.neurons) {
      print("[${n.value}]");
    }
  }

  MultiLayerPerceptron.train(mlp, dataset, 1000, 0.4, bias: 0);

  print("============");
  print("Output after training");
  print("============");
  for (Pair i in dataset.pairs) {
    MultiLayerPerceptron.forward(mlp, i.inputData, bias: 0);
    print("Inputs:");
    for (Neuron n in mlp.layers[0]!.neurons) {
      print("[${n.value}]");
    }
    print("Outputs:");
    for (Neuron n in mlp.layers[2]!.neurons) {
      print("[${n.value}]");
    }
  }

  /*print("============");
  print("Output after training");
  print("============");
  for (var i in dataset.pairs) {
    MultiLayerPerceptron.forward(mlp, i.inputData, bias: 0);
    print(
        'inputs: ${mlp.layers[0]!.neurons[0].value}, ${mlp.layers[0]!.neurons[1].value}');
    print('output: ${mlp.layers[2]!.neurons[0].value}');
  }*/
}

Future<Dataset> loadDataset() async {
  var dataset = Dataset(); // Empty Dataset

  final data = await XMLUtils.loadData();
  for (List<dynamic> line in data) {
    print(line[0]);
    if (line[2].toString() == "treino") {
      var value = line[0];
      var input = <double>[];
      var split = value.toString().split('');

      for (String splitted in split) {
        var digit = int.tryParse(splitted);
        input.add(digit!.toDouble());
      }

      // Saídas esperadas de acordo com o dataset
      var expectedOutput = getExpectedOutput(line[1].toString());

      // Preenche o dataset com os dados de entrada e saída
      dataset.pairs.add(Pair(input, expectedOutput));
    }
  }

  return dataset;
}

Future<Dataset> loadTestDataset() async {
  var dataset = Dataset(); // Empty Dataset

  final data = await XMLUtils.loadData();
  for (List<dynamic> line in data) {
    if (line[2].toString() == "teste") {
      var value = line[0];
      var input = <double>[];
      var split = value.toString().split('');

      for (String splitted in split) {
        var digit = int.tryParse(splitted);
        input.add(digit!.toDouble());
      }

      // Saídas esperadas de acordo com o dataset
      var expectedOutput = getExpectedOutput(line[1].toString());

      // Preenche o dataset com os dados de entrada e saída
      dataset.pairs.add(Pair(input, expectedOutput));
    }
  }

  return dataset;
}

List<double> getExpectedOutput(String output) {
  if (output == "primeiro numero") {
    return [0, 1, 0];
  } else if (output == "segundo numero") {
    return [0, 0, 1];
  }
  return [1, 0, 0];
}

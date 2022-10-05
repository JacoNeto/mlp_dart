import 'dart:convert';
import 'dart:io';

import 'package:csv/csv.dart';
import 'package:mlpdart/mlp/dataset.dart';
import 'package:mlpdart/mlp/layer.dart';
import 'package:mlpdart/mlp/mlp.dart';
import 'package:mlpdart/mlp/neuron.dart';
import 'package:mlpdart/csv/csv_utils.dart';

Future<void> mlpdart() async {
  // Set Min and Max weight value for all neurons
  Neuron.setRange(-1, 1);

  // Create a Neural Network with 3 Layers
  MultiLayerPerceptron mlp =
      MultiLayerPerceptron(3); // 1 input + 1 hidden + 1 output
  // No need to add input layer, it will be added from trainingDataset automatically
  mlp.layers[1] = Layer.hidden(
      8, 20); // Hidden layer / 8 neurons each have 20 weights (comlpections)
  mlp.layers[2] = Layer.hidden(
      3, 8); // Output layer / 3 neuron with 8 weights (comlpections)

  // Create the training trainingDataset
  Dataset trainingDataset = await loadDataset();
  // Create the testing trainingDataset
  Dataset testingDataset = await loadTestDataset();

  print("\n*********************************************");
  print("****Valores de treino na primeira iteração***");
  print("*********************************************");

  for (Pair i in trainingDataset.pairs) {
    MultiLayerPerceptron.forwardpropagation(mlp, i.inputData, bias: 0);

    var str = "";
    print("\nInputs:");
    for (Neuron n in mlp.layers[0]!.neurons) {
      str += n.value!.toStringAsFixed(0);
    }
    print(str);

    str = "";
    print("--------------------");
    print("Outputs:");
    print("?                  5                     6");
    int aux = 0;
    var greaterIndex = 0;
    double greater = mlp.layers[2]!.neurons[0].value!;
    for (Neuron n in mlp.layers[2]!.neurons) {
      if (n.value! > greater) {
        greater = n.value!;
        greaterIndex = aux;
      }
      str += "${n.value!.toString()} ";
      aux++;
    }
    print(str);

    rightWrong(i.outputData, greaterIndex);
  }

  MultiLayerPerceptron.train(mlp, trainingDataset, 1000, 0.4, bias: 0);

  print("\n\n\n***********************");
  print("****Valores de Teste***");
  print("***********************\n");
  for (Pair i in testingDataset.pairs) {
    var str = "";
    MultiLayerPerceptron.forwardpropagation(mlp, i.inputData, bias: 0);
    print("\nInputs:");
    for (Neuron n in mlp.layers[0]!.neurons) {
      str += n.value!.toStringAsFixed(0);
    }
    print(str);

    str = "";
    print("--------------------");
    print("Outputs:");
    print("?                  5                     6");

    int aux = 0;
    var greaterIndex = 0;
    double greater = mlp.layers[2]!.neurons[0].value!;
    for (Neuron n in mlp.layers[2]!.neurons) {
      if (n.value! > greater) {
        greater = n.value!;
        greaterIndex = aux;
      }
      str += "${n.value!.toString()} ";
      aux++;
    }
    print(str);

    rightWrong(i.outputData, greaterIndex);
  }
}

Future<Dataset> loadDataset() async {
  var trainingDataset = Dataset(); // Empty Dataset

  final data = await XMLUtils.loadData();
  for (List<dynamic> line in data) {
    if (line[2].toString() == "treino") {
      var value = line[0];
      var input = <double>[];
      var split = value.toString().split('');

      for (String splitted in split) {
        var digit = int.tryParse(splitted);
        input.add(digit!.toDouble());
      }

      // Saídas esperadas de acordo com o trainingDataset
      var expectedOutput = getExpectedOutput(line[1].toString());

      // Preenche o trainingDataset com os dados de entrada e saída
      trainingDataset.pairs.add(Pair(input, expectedOutput));
    }
  }

  return trainingDataset;
}

Future<Dataset> loadTestDataset() async {
  var trainingDataset = Dataset(); // Empty Dataset

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

      // Saídas esperadas de acordo com o trainingDataset
      var expectedOutput = getExpectedOutput(line[1].toString());

      // Preenche o trainingDataset com os dados de entrada e saída
      trainingDataset.pairs.add(Pair(input, expectedOutput));
    }
  }

  return trainingDataset;
}

List<double> getExpectedOutput(String output) {
  if (output == "primeiro numero") {
    return [0, 1, 0];
  } else if (output == "segundo numero") {
    return [0, 0, 1];
  }
  return [1, 0, 0];
}

void rightWrong(List<double> expected, int index) {
  print("expected: $expected");
  if (expected[1] == 1) {
    if (index == 1) {
      print("ACERTOU");
    } else {
      print("ERROU");
    }
  } else if (expected[2] == 1) {
    if (index == 2) {
      print("ACERTOU");
    } else {
      print("ERROU");
    }
  } else {
    if (index == 0) {
      print("ACERTOU");
    } else {
      print("ERROU");
    }
  }
}

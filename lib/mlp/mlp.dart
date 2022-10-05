import 'dataset.dart';
import 'layer.dart';
import 'package:mlpdart/math/function_utils.dart' as fu;

import 'neuron.dart';

// Nessa classe ocorrem todos os processamentos do MLP
class MultiLayerPerceptron {
  // Camadas do MLP
  List<Layer?> layers = [];
  // Número de camadas
  int size;

  // O construtor inicializa o número de camadas informado no início
  MultiLayerPerceptron(this.size) {
    layers = List<Layer?>.filled(size, null);
  }

  static void forwardpropagation(MultiLayerPerceptron mlp, List<double> inputs,
      {double bias = 0}) {
    // Bring the inputs into the input layer
    mlp.layers[0] = Layer.entry(inputs);

    // Forward propagation
    for (int i = 1; i < mlp.layers.length; i++) {
      // Starts from 1st hidden layer
      for (int j = 0; j < mlp.layers[i]!.neurons.length; j++) {
        double net = 0;
        for (int k = 0; k < mlp.layers[i - 1]!.neurons.length; k++) {
          net += mlp.layers[i - 1]!.neurons[k].value! *
                  mlp.layers[i]!.neurons[j].weights[k] +
              bias;
        }
        mlp.layers[i]!.neurons[j].value = fu.sigmoid(net);
      }
    }
  }

  static void backpropagation(
      MultiLayerPerceptron mlp, double learningRate, Pair datas) {
    int numberLayers = mlp.layers.length;
    int outputLayerIndex = numberLayers - 1;

    // Atualiza as camadas de saída para cada saída e peso
    for (int i = 0; i < mlp.layers[outputLayerIndex]!.neurons.length; i++) {
      double output = mlp.layers[outputLayerIndex]!.neurons[i].value!;
      double target = datas.outputData[i];
      double derivative = output - target;
      double delta = derivative * (output * (1 - output));
      mlp.layers[outputLayerIndex]!.neurons[i].gradient = delta;

      for (int j = 0;
          j < mlp.layers[outputLayerIndex]!.neurons[i].weights.length;
          j++) {
        double previousOutput =
            mlp.layers[outputLayerIndex - 1]!.neurons[j].value!;
        double error = delta * previousOutput;
        mlp.layers[outputLayerIndex]!.neurons[i].previousWeights[j] =
            mlp.layers[outputLayerIndex]!.neurons[i].weights[j] -
                learningRate * error;
      }
    }

    // Atualiza todas as camadas ocultas segiuntas, para cada neurônio na camada
    // e seus pesos
    for (int i = outputLayerIndex - 1; i > 0; i--) {
      // Backward
      for (int j = 0; j < mlp.layers[i]!.neurons.length; j++) {
        double output = mlp.layers[i]!.neurons[j].value!;
        double gradientSum = sumGradient(mlp, j, i + 1);
        double delta = (gradientSum) * (output * (1 - output));
        mlp.layers[i]!.neurons[j].gradient = delta;

        for (int k = 0; k < mlp.layers[i]!.neurons[j].weights.length; k++) {
          // And for all their weights
          double previousOutput = mlp.layers[i - 1]!.neurons[k].value!;
          double error = delta * previousOutput;
          mlp.layers[i]!.neurons[j].previousWeights[k] =
              mlp.layers[i]!.neurons[j].weights[k] - learningRate * error;
        }
      }
    }

    // Update all the weights
    for (int i = 0; i < mlp.layers.length; i++) {
      for (int j = 0; j < mlp.layers[i]!.neurons.length; j++) {
        mlp.layers[i]!.neurons[j].setWeights();
      }
    }
  }

// Soma dos gradientes de uma dada camada
  static double sumGradient(
      MultiLayerPerceptron mlp, int neuronIndex, int layerIndex) {
    double gradientSum = 0;
    Layer currentLayer = mlp.layers[layerIndex]!;
    for (int i = 0; i < currentLayer.neurons.length; i++) {
      Neuron currentNeuron = currentLayer.neurons[i];
      gradientSum +=
          currentNeuron.weights[neuronIndex] * currentNeuron.gradient;
    }
    return gradientSum;
  }

// Método de treino com o MLP
  static void train(MultiLayerPerceptron mlp, Dataset dataset, int iterations,
      double learningRate,
      {double bias = 0}) {
    for (int i = 0; i < iterations; i++) {
      for (int j = 0; j < dataset.getLength(); j++) {
        forwardpropagation(mlp, dataset.pairs[j].inputData, bias: bias);
        backpropagation(mlp, learningRate, dataset.pairs[j]);
      }
    }
  }
}
